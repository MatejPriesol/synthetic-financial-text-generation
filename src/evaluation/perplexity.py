from pathlib import Path
from src.data_loader import load_synthetic_fire_data, load_processed_fire_data, load_processed_markdown_data

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np


def evaluate_perplexity(sentences):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device.upper()}")

    # Load model (for now GPT-2, later I will try Llama 3.1.-8B)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    model.eval()  # Set to evaluation mode

    scored_sentences = []

    with torch.no_grad():  # Disable gradient tracking to save memory
        for text in sentences:
            # Skip empty sentences (shouldn't be the case)
            if not text.strip():
                print("Empty sentence!")
                continue

            # Tokenize the sentence
            encodings = tokenizer(text, return_tensors='pt').to(device)
            input_ids = encodings.input_ids

            # Skip just in case there's one word (shouldn't be the case)
            if input_ids.size(1) < 2:
                print("One word sentence!")
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

    print(f"Average Perplexity: {avg_ppl:.2f}")
    print(f"Median Perplexity:  {median_ppl:.2f}\n")

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

    evaluate_perplexity(sentences)

if __name__ == "__main__":
    main()
