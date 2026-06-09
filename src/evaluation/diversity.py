import torch
import nltk
import argparse
import json
import random

from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from collections import Counter

from src.data_loader import load_data_for_eval


def evaluate_self_bleu(sentences, num_hypothesis_to_test=200, verbose=False):
    tokenized_sentences = [nltk.word_tokenize(text.lower()) for text in sentences]
    num_sentences = len(tokenized_sentences)

    if num_sentences < 2:
        return {"self_bleu_2": 0, "self_bleu_3": 0, "self_bleu_4": 0}

    # Smoothing is required for sentence-level BLEU to avoid 0 scores on short sentences
    chencherry = SmoothingFunction()

    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []

    # Limit the number of hypotheses we test to speed up the loop
    actual_test_size = min(num_hypothesis_to_test, num_sentences)

    if verbose:
        print(f"Calculating Self-BLEU for {actual_test_size} sentences (This may take a moment)...")

    hypothesis_indices = random.sample(range(num_sentences), actual_test_size)

    for i in hypothesis_indices:
        hypothesis = tokenized_sentences[i]

        # All available references except the current sentence
        references = tokenized_sentences[:i] + tokenized_sentences[i+1:]

        # Calculate BLEU-2, BLEU-3, and BLEU-4
        b2 = sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
        b3 = sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
        b4 = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

        bleu_2_scores.append(b2)
        bleu_3_scores.append(b3)
        bleu_4_scores.append(b4)

    avg_bleu_2 = sum(bleu_2_scores) / actual_test_size
    avg_bleu_3 = sum(bleu_3_scores) / actual_test_size
    avg_bleu_4 = sum(bleu_4_scores) / actual_test_size

    if verbose:
        print(f"Average Self-BLEU-2: {avg_bleu_2:.4f}")
        print(f"Average Self-BLEU-3: {avg_bleu_3:.4f}")
        print(f"Average Self-BLEU-4: {avg_bleu_4:.4f}\n")

    return {
        "self_bleu_2": round(avg_bleu_2, 4),
        "self_bleu_3": round(avg_bleu_3, 4),
        "self_bleu_4": round(avg_bleu_4, 4)
    }


def evaluate_ngrams_words(sentences, n=3, top_k=10, verbose=False):

    all_ngrams = []

    for text in sentences:
        tokens = nltk.word_tokenize(text.lower())
        sentence_ngrams = list(ngrams(tokens, n))
        all_ngrams.extend(sentence_ngrams)

    # Count frequencies
    ngram_counts = Counter(all_ngrams)
    total_ngrams = sum(ngram_counts.values())
    unique_ngrams = len(ngram_counts)

    # Calculate Unique N-Gram Ratio
    uniqueness_ratio = (unique_ngrams / total_ngrams) * 100 if total_ngrams > 0 else 0

    if verbose:
        print(f"{str(top_k) + ' Most Frequent ' + str(n) + '-Grams':<30} | {'Count'}")
        print("-" * 40)

        for phrase, count in ngram_counts.most_common(top_k):
            phrase_str = " ".join(phrase)
            print(f"{phrase_str:<30} | {count:>4}")

    return {
        f"total_{n}grams": total_ngrams,
        f"unique_{n}grams": unique_ngrams,
        f"{n}gram_uniqueness_ratio": round(uniqueness_ratio, 2)
    }

def evaluate_semantic_similarity(sentences, threshold=0.9, verbose=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embeddings, embeddings)

    num_of_sentences = len(sentences)

    # 1) Overall pairwise similarity:
    # Get the row and column indices for the upper triangle (offset skips the diagonal)
    row_idx, col_idx = torch.triu_indices(num_of_sentences, num_of_sentences, offset=1, device=device)

    # Get the scores
    pairwise_scores = scores[row_idx, col_idx]
    avg_score = pairwise_scores.mean().item()

    sorted_scores, sorted_indices = torch.sort(pairwise_scores, descending=True)
    sorted_scores_list = sorted_scores.tolist()
    sorted_rows = row_idx[sorted_indices].tolist()
    sorted_cols = col_idx[sorted_indices].tolist()

    if verbose:
        print("Most similar non-identical sentences:")

    count = 0
    iterated = 0

    for i in range(len(sorted_scores_list)):
        score = sorted_scores_list[i]

        if score >= threshold:
            iterated += 1
            continue

        idx1 = sorted_rows[i]
        idx2 = sorted_cols[i]

        if verbose:
            print(f"Pair {count + 1} - Score: {score:.3f}, idx1: {idx1}, idx2: {idx2}")
            print(f"  A: {sentences[idx1]}")
            print(f"  B: {sentences[idx2]}\n")

        count += 1
        if count >= 10:
            break

    # 2) Closest match similarity:
    scores.fill_diagonal_(-1)  # Ignore self-matching
    max_scores, _ = torch.max(scores, dim=1)
    max_scores_list = max_scores.tolist()

    duplicates_percentage = sum(1 for s in max_scores_list if s >= 0.98) * 100 / len(max_scores_list)
    very_similar_percentage = sum(1 for s in max_scores_list if 0.85 <= s < 0.98) * 100 / len(max_scores_list)
    different_percentage = sum(1 for s in max_scores_list if s < 0.85) * 100 / len(max_scores_list)

    avg_closest = max_scores.mean().item()
    median_closest = max_scores.median().item()

    return {
        "all_pairs_similarity": {
            "avg_pairwise_sim": round(avg_score, 3),
            f"percentage_above_{threshold}": round(100 * iterated / len(sorted_scores_list), 5),
        },
        "closest_match_similarity": {
            "avg_closest_match": round(avg_closest, 3),
            "median_closest_match": round(median_closest, 3),
            "duplicates_percentage": round(duplicates_percentage, 3),
            "very_similar_percentage": round(very_similar_percentage, 3),
            "different_percentage": round(different_percentage, 3)
        }
    }


def run_all_diversity(sentences, verbose=False):
    module_results = {}

    semantic_res = evaluate_semantic_similarity(sentences, verbose=verbose)
    module_results.update(semantic_res)

    module_results["ngrams"] = evaluate_ngrams_words(sentences, verbose=verbose)
    module_results["self_bleu"] = evaluate_self_bleu(sentences, verbose=verbose)

    return module_results

def main():
    parser = argparse.ArgumentParser(description="Diversity Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--semantic", action="store_true", help="Run only semantic analysis")
    parser.add_argument("--ngrams", action="store_true", help="Run only n-gram analysis")
    parser.add_argument("--self_bleu", action="store_true", help="Run only Self-BLEU analysis")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.semantic,
        args.ngrams,
        args.self_bleu
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected diversity evaluations...\n")
        if args.semantic:
            local_results["semantic"] = evaluate_semantic_similarity(sentences, verbose=args.verbose)
        if args.ngrams:
            local_results["ngrams"] = evaluate_ngrams_words(sentences, verbose=args.verbose)
        if args.self_bleu:
            local_results["self_bleu"] = evaluate_self_bleu(sentences, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all diversity evaluations by default...\n")
        local_results = run_all_diversity(sentences, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()
