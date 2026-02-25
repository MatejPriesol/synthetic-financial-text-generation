import nltk
from nltk.util import ngrams

from collections import Counter
from pathlib import Path
from src.data_loader import load_synthetic_fire_data

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


def main():
    synthetic_data_path = Path("data") / "03_synthetic" / "few_shot_650_samples_4_examples.json"
    synthetic_data = load_synthetic_fire_data(synthetic_data_path)

    sentences = [sample.sentence for sample in synthetic_data]
    evaluate_ngrams_words(sentences, n=3)

if __name__ == "__main__":
    main()
