import torch
import nltk
from nltk.util import ngrams

from sentence_transformers import SentenceTransformer, util
from collections import Counter
from pathlib import Path
from src.data_loader import load_synthetic_fire_data, load_processed_fire_data, load_processed_markdown_data


def evaluate_ngrams_words(sentences, n=3, top_k=10):

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

    print(f"Total {n}-grams generated: {total_ngrams}")
    print(f"Unique {n}-grams:        {unique_ngrams}")
    print(f"Uniqueness Ratio:      {uniqueness_ratio:.1f}%\n")

    print(f"{'Most Frequent ' + str(n) + '-Grams':<30} | {'Count'}")
    print("-" * 40)

    for phrase, count in ngram_counts.most_common(top_k):
        phrase_str = " ".join(phrase)
        print(f"{phrase_str:<30} | {count:>4}")


def evaluate_semantic_similarity(sentences):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device.upper()}")

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(embeddings, embeddings)

    pairs = []
    num_of_sentences = len(sentences)

    for i in range(num_of_sentences):
        for j in range(i+1, num_of_sentences):
            pairs.append({'id': [i, j], 'score': scores[i, j].item()})

    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)

    avg_score = sum(p['score'] for p in pairs) / len(pairs) if pairs else 0

    print(f"Average Pairwise Similarity: {avg_score:.3f}\n")

    """
    print("Most similar sentences:")
    for i in range(min(10, len(pairs))):
        idx1, idx2 = pairs[i]['id']
        score = pairs[i]['score']
        print(f"Pair {i + 1} - Score: {score:.3f}, idx1: {idx1}, idx2: {idx2}")
        print(f"  A: {sentences[idx1]}")
        print(f"  B: {sentences[idx2]}\n")
    """

    print("Most similar non identical sentences:")

    count = 0
    iterated = 0
    for pair in pairs:
        score = pair['score']

        if score >= 0.9:
            iterated += 1
            continue

        idx1, idx2 = pair['id']
        print(f"Pair {count + 1} - Score: {score:.3f}, idx1: {idx1}, idx2: {idx2}")
        print(f"  A: {sentences[idx1]}")
        print(f"  B: {sentences[idx2]}\n")

        count += 1
        if count >= 10:
            break

    print(f"Iterated over {iterated+count} pairs:")

    print(f"Percentage of sentences with score >= 0.9: {100*iterated/len(pairs):.5f}%")

    return scores



def main():
    synthetic_data_path = Path("data") / "03_synthetic" / "few_shot_650_samples_4_examples.json"
    #synthetic_data_path = Path("data") / "03_synthetic" / "few_shot_100_examples.json"
    synthetic_data = load_synthetic_fire_data(synthetic_data_path)

    fire_data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    fire_data = load_processed_fire_data(fire_data_path)

    elekta_data_path = Path("data") / "02_processed" / "elekta_processed.json"
    elekta_data = load_processed_markdown_data(elekta_data_path)
    vestas_data_path = Path("data") / "02_processed" / "vestas_processed.json"
    vestas_data = load_processed_markdown_data(vestas_data_path)

    # sentences = [sample.sentence for sample in synthetic_data]
    # sentences = [sample.sentence for sample in fire_data]
    # sentences = [sample.sentence for sample in elekta_data]
    sentences = [sample.sentence for sample in vestas_data]

    #evaluate_ngrams_words(sentences, n=3)
    evaluate_semantic_similarity(sentences)

if __name__ == "__main__":
    main()
