import torch
import nltk
import argparse
import json

from nltk.util import ngrams
from sentence_transformers import SentenceTransformer, util
from collections import Counter

from src.data_loader import load_data_for_eval


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
    #print(f"Using {device.upper()}")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embeddings, embeddings)

    num_of_sentences = len(sentences)

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


    return {
        "avg_pairwise_sim": round(avg_score, 3),
        f"percentage_above_{threshold}": round(100 * iterated / len(sorted_scores_list), 5)
    }, scores


def run_all_diversity(sentences, verbose=False):
    module_results = {}

    module_results["semantic"], _ = evaluate_semantic_similarity(sentences, verbose=verbose)
    module_results["ngrams"] = evaluate_ngrams_words(sentences, verbose=verbose)

    return module_results

def main():
    parser = argparse.ArgumentParser(description="Diversity Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--semantic", action="store_true", help="Run only semantic analysis")
    parser.add_argument("--ngrams", action="store_true", help="Run only n-gram analysis")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.semantic,
        args.ngrams
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected diversity evaluations...\n")
        if args.semantic:
            local_results["semantic"], _ = evaluate_semantic_similarity(sentences, verbose=args.verbose)
        if args.ngrams:
            local_results["ngrams"] = evaluate_ngrams_words(sentences, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all diversity evaluations by default...\n")
        local_results = run_all_diversity(sentences, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()
