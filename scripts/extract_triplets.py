import json
import torch
import argparse
import sys

from pathlib import Path
from transformers import pipeline
from dataclasses import asdict

from src.data_loader import load_processed_markdown_data, Triplet


def parse_rebel_output(text):
    """
    Official parsing function from the Babelscape/rebel model GitHub page (modified subj/obj names)
    Taken from: https://github.com/Babelscape/rebel
    TODO: Include citation in paper!
    """
    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'

    # Clean out the special generation tokens
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'relation': relation.strip(), 'object': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'subject': subject.strip(), 'relation': relation.strip(), 'object': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    if subject != '' and relation != '' and object_ != '':
        triplets.append({'subject': subject.strip(), 'relation': relation.strip(), 'object': object_.strip()})

    return triplets


def process_rebel_extraction(input_filepath, output_filepath, batch_size=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device.upper()}")

    triplet_extractor = pipeline(
        'text2text-generation',
        model='Babelscape/rebel-large',
        tokenizer='Babelscape/rebel-large',
        device=device,
        batch_size=batch_size
    )

    # Load data
    print(f"Loading data from {input_filepath}...")
    samples = load_processed_markdown_data(input_filepath)

    print(f"Extracting triplets for {len(samples)} sentences...")

    sentences = [sample.sentence for sample in samples]

    results = triplet_extractor(sentences, truncation=True, max_length=256, return_tensors=True,
        return_text=False)

    # Re-attach the extracted triplets
    for i, res in enumerate(results):
        raw_tokens = res['generated_token_ids']

        raw_text = triplet_extractor.tokenizer.decode(
            raw_tokens,
            skip_special_tokens=False
        )

        parsed_triplets = parse_rebel_output(raw_text)

        # Convert the dict output to Triplet object
        samples[i].triplets = [Triplet(**t) for t in parsed_triplets]

    # Save the data
    data_to_save = [asdict(sample) for sample in samples]

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset to extract triplets from")

    parser.add_argument(
        "--data",
        type=str,
        choices=['elekta', 'vestas'],
        help="Choose which dataset to process. Please specify --data flag."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()


    input_filepath = Path("data") / "02_processed" / f"{args.data}_processed.json"
    output_filepath = Path("data") / "02_processed" / f"{args.data}_processed_w_triplets.json"

    process_rebel_extraction(input_filepath, output_filepath)