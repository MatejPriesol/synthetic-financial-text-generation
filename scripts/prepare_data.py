import json
import argparse
import sys

from pathlib import Path
from dataclasses import asdict
from src.data_loader import load_fire_data, load_fire_relations, load_markdown_data


def prepare_fire_data():
    raw_path_data = Path("data") / "01_raw" / "fire_train.json"

    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path_data = processed_dir / "fire_train_processed.json"

    print("Loading and parsing raw FIRE data")
    processed_data = load_fire_data(raw_path_data)

    # Convert to dict
    data_to_save = [asdict(sample) for sample in processed_data]

    # Save into file
    with open(output_path_data, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"Data saved to {output_path_data}")


def prepare_fire_relations():
    raw_path_relations = Path("data") / "01_raw" / "fire_types.json"

    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path_relations = processed_dir / "fire_types_processed.json"

    print("Loading and parsing FIRE relations")
    relations = load_fire_relations(raw_path_relations)

    with open(output_path_relations, 'w', encoding='utf-8') as f:
        json.dump(relations, f, indent=4)
    print(f"Relations saved to {output_path_relations}")


def prepare_markdown_data():
    raw_path_elekta = Path("data") / "01_raw" / "Elekta_Annual_Report_2023.txt"

    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path_data = processed_dir / "elekta_processed.json"

    print("Loading and parsing raw Elekta data")
    processed_data = load_markdown_data(filepath=raw_path_elekta, name="elekta")

    # Convert to dict
    data_to_save = [asdict(sample) for sample in processed_data]

    # Save into file
    with open(output_path_data, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f"Data saved to {output_path_data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare datasets for the pipeline.")

    parser.add_argument(
        "--data",
        type=str,
        choices=['fire', 'relations', 'elekta', 'all'],
        help="Choose which dataset to process. Please specify --data flag."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.data in ['fire', 'all']:
        prepare_fire_data()

    if args.data in ['relations', 'all']:
        prepare_fire_relations()

    if args.data in ['elekta', 'all']:
        prepare_markdown_data()