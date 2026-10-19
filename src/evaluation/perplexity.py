"""
Perplexity Evaluation Module.

This script calculates the perplexity of generated text using a pre-trained GPT-2 model.
It evaluates how "surprising" or predictable the synthetic sentences are. Higher perplexity
generally indicates less predictable/more diverse text, while very low perplexity suggests
repetitive or highly generic generation.

Note: This evaluation is excluded from the main script and is not run automatically.
Local evaluation is still possible within this script.
"""

import torch
import argparse
import json
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.data_loader import load_data_for_eval

def evaluate_perplexity(sentences, verbose=False):
    """
    Evaluates the perplexity of a list of sentences using GPT-2.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()  # Set to evaluation mode

    scored_sentences = []

    with torch.no_grad():  # Disable gradient tracking to save memory
        for text in sentences:
            # Skip empty sentences (shouldn't be the case)
            if not text.strip() and verbose:
                continue

            # Tokenize the sentence
            encodings = tokenizer(text, return_tensors='pt').to(device)
            input_ids = encodings.input_ids

            # Skip just in case there's one word (shouldn't be the case)
            if (input_ids.size(1) < 2) and verbose:
                continue

            # Negative log-likelihood
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            # Exponential of the loss
            ppl = torch.exp(loss).item()

            scored_sentences.append({'text': text, 'perplexity': ppl})

    # Sort the list from Lowest Perplexity (predictable) to Highest (surprising)
    scored_sentences.sort(key=lambda x: x['perplexity'])

    # Calculate Average
    all_scores = [s['perplexity'] for s in scored_sentences]
    avg_ppl = np.mean(all_scores)
    median_ppl = np.median(all_scores)

    if verbose:
        print("Low perplexity:")
        for i in range(5):
            print(f"Score: {scored_sentences[i]['perplexity']:.2f} | {scored_sentences[i]['text']}")

        print("\nMedian perplexity:")
        mid_idx = len(scored_sentences) // 2
        for i in range(mid_idx - 2, mid_idx + 3):
            print(f"Score: {scored_sentences[i]['perplexity']:.2f} | {scored_sentences[i]['text']}")

        print("\nHigh perplexity:")
        for i in range(1, 6):
            print(f"Score: {scored_sentences[-i]['perplexity']:.2f} | {scored_sentences[-i]['text']}")

    return {
        "avg_perplexity": round(avg_ppl, 3),
        "median_perplexity": round(median_ppl, 3),
    }


def run_all_perplexity(sentences, verbose=False):
    return {"perplexity": evaluate_perplexity(sentences, verbose=verbose)}


def main():
    parser = argparse.ArgumentParser(description="Perplexity Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--perplexity", action="store_true", help="Run only perplexity analysis")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.perplexity
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected perplexity evaluations...\n")
        if args.perplexity:
            local_results["perplexity"] = evaluate_perplexity(sentences, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all perplexity evaluations by default...\n")
        local_results = run_all_perplexity(sentences, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()
