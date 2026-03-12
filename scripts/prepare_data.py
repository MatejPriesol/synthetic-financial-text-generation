import json
import argparse
import sys

from pathlib import Path
from dataclasses import asdict
from src.data_loader import load_fire_data, load_fire_types, load_markdown_data, load_rebel_data


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


def prepare_fire_types():
    raw_path_types = Path("data") / "01_raw" / "fire_types.json"

    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path_relations = processed_dir / "fire_relation_types_processed.json"

    print("Loading and parsing FIRE relations")
    relations = load_fire_types(raw_path_types, "relations")

    with open(output_path_relations, 'w', encoding='utf-8') as f:
        json.dump(relations, f, indent=4)
    print(f"Relations saved to {output_path_relations}")

    output_path_entities = processed_dir / "fire_entity_types_processed.json"

    print("Loading and parsing FIRE entities")
    entities = load_fire_types(raw_path_types, "entities")

    with open(output_path_entities, 'w', encoding='utf-8') as f:
        json.dump(entities, f, indent=4)
    print(f"Entities saved to {output_path_entities}")


def prepare_markdown_data(name, filename):
    raw_path = Path("data") / "01_raw" / f"{filename}"

    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path_data = processed_dir / f"{name}_processed.json"

    print(f"Loading and parsing raw {name} data")
    processed_data = load_markdown_data(filepath=raw_path, name=name)

    # Convert to dict
    data_to_save = [asdict(sample) for sample in processed_data]

    # Save into file
    with open(output_path_data, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f"Data saved to {output_path_data}")


def prepare_rebel_data(name, filename):
    raw_path = Path("data") / "01_raw" / filename

    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path_data = processed_dir / f"{name}_processed.json"

    print(f"Loading and parsing raw {name} data")
    processed_data = load_rebel_data(filepath=raw_path, name=name)

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
        choices=['fire', 'types', 'elekta', 'vestas', 'rebel', 'all'],
        help="Choose which dataset to process. Please specify --data flag."
    )

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.data in ['fire', 'all']:
        prepare_fire_data()

    if args.data in ['types', 'all']:
        prepare_fire_types()

    if args.data in ['elekta', 'all']:
        prepare_markdown_data(name="elekta", filename="Elekta_Annual_Report_2023.txt")

    if args.data in ['vestas', 'all']:
        prepare_markdown_data(name="vestas", filename="Vestas_Annual_Report_2024.txt")

    if args.data in ['rebel', 'all']:
        prepare_rebel_data(name="rebel", filename="rebel_val.jsonl")