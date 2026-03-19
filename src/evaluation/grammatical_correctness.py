import torch
import argparse
import json
import numpy as np

from transformers import pipeline

from src.data_loader import load_data_for_eval


def evaluate_grammatical_correctness(sentences, verbose=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    if verbose:
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

    return {
        "avg_score": round(avg_score, 3),
        "median_score": round(median_score, 3),
    }


def run_all_grammar(sentences, verbose=False):
    return {"grammar": evaluate_grammatical_correctness(sentences, verbose=verbose)}


def main():
    parser = argparse.ArgumentParser(description="Grammatical Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--grammar", action="store_true", help="Run only grammar analysis")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.grammar
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected grammatical evaluations...\n")
        if args.grammar:
            local_results["grammar"] = evaluate_grammatical_correctness(sentences, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all grammatical evaluations by default...\n")
        local_results = run_all_grammar(sentences, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()
