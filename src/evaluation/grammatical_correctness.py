from pathlib import Path
from src.data_loader import load_synthetic_fire_data, load_processed_fire_data, load_processed_markdown_data, load_rebel_sample

import torch
from transformers import pipeline
import numpy as np

def evaluate_grammatical_correctness(sentences):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device.upper()}")

    grammar_checker = pipeline(
        "text-classification",
        model="textattack/roberta-base-CoLA",
        device=device,
        batch_size=32
    )

    scored_sentences = []
    results = grammar_checker(
        sentences,
        truncation=True,
        max_length=512
    )

    for sentence, res in zip(sentences, results):

        is_acceptable = True if res['label'] == 'LABEL_1' else False
        score = res['score']

        scored_sentences.append({
            'sentence': sentence,
            'grammatically correct': is_acceptable,
            'score': score if is_acceptable else (1.0 - score)    # Since score is the confidence, invert for ungrammatical sentences
        })

    # Sort the list from worst to best
    scored_sentences.sort(key=lambda x: x['score'])

    # Calculate Average
    all_scores = [s['score'] for s in scored_sentences]
    avg_score = np.mean(all_scores)
    median_score = np.median(all_scores)

    print(f"Average Grammatical Score: {avg_score:.2f}")
    print(f"Median Grammatical Score:  {median_score:.2f}\n")

    print("Worst scores:")
    for i in range(5):
        print(f"Score: {scored_sentences[i]['score']:.2f} | {scored_sentences[i]['sentence']}")

    print("\nMedian scores:")
    mid_idx = len(scored_sentences) // 2
    for i in range(mid_idx - 2, mid_idx + 3):
        print(f"Score: {scored_sentences[i]['score']:.2f} | {scored_sentences[i]['sentence']}")

    print("\nBest scores:")
    for i in range(1, 6):
        print(f"Score: {scored_sentences[-i]['score']:.2f} | {scored_sentences[-i]['sentence']}")

def main():
    synthetic_data_path = Path("data") / "03_synthetic" / "few_shot_650_samples_4_examples.json"
    #synthetic_data_path = Path("data") / "03_synthetic" / "ontology_baseline_2_percent.json"
    synthetic_data_path = Path("data") / "03_synthetic" / "ontology_2_percent_half_fixed.json"
    synthetic_data = load_synthetic_fire_data(synthetic_data_path)

    fire_data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    fire_data = load_processed_fire_data(fire_data_path)

    elekta_data_path = Path("data") / "02_processed" / "elekta_processed.json"
    elekta_data = load_processed_markdown_data(elekta_data_path)
    vestas_data_path = Path("data") / "02_processed" / "vestas_processed.json"
    vestas_data = load_processed_markdown_data(vestas_data_path)

    rebel_data_path = Path("data") / "02_processed" / "rebel_processed.json"
    rebel_data = load_rebel_sample(rebel_data_path, size=10000)

    sentences = [sample.sentence for sample in synthetic_data]
    #sentences = [sample.sentence for sample in fire_data]
    #sentences = [sample.sentence for sample in elekta_data]
    #sentences = [sample.sentence for sample in vestas_data]
    #sentences = [sample.sentence for sample in rebel_data]

    evaluate_grammatical_correctness(sentences)

if __name__ == "__main__":
    main()
