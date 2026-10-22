import json
import argparse
import sys

from pathlib import Path
from dataclasses import asdict
from src.data_loader import load_fire_data, load_fire_types, load_markdown_data, load_rebel_data

def prepare_fire_data(output_filename="fire_all_processed.json"):
    raw_dir = Path("data") / "01_raw"
    processed_dir = Path("data") / "02_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_files = ["fire_train.json", "fire_dev.json", "fire_test.json"]
    combined_data = []

    for filename in raw_files:
        file_path = raw_dir / filename
        if not file_path.exists():
            print(f"[WARNING] File {file_path} not found. Skipping...")
            continue

        print(f"Loading {filename}...")
        split_data = load_fire_data(file_path)
        combined_data.extend(split_data)

    if not combined_data:
        print("[WARNING] No FIRE data was loaded. Skipping processing.")
        return

    print(f"\nTotal loaded samples across splits: {len(combined_data)}")

    # Check for duplicate IDs
    all_ids = [sample.id for sample in combined_data]
    unique_ids = set(all_ids)
    if len(all_ids) != len(unique_ids):
        print(f"Found {len(all_ids) - len(unique_ids)} duplicate IDs across splits")
    else:
        print("No duplicate IDs found across splits.")

    # Sort by original ID
    combined_data.sort(key=lambda sample: sample.id)

    # Save processed output
    data_to_save = [asdict(sample) for sample in combined_data]
    output_path = processed_dir / output_filename

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"Data saved to {output_path}\n")


def prepare_fire_types():
    raw_path_types = Path("data") / "01_raw" / "fire_types.json"

    if not raw_path_types.exists():
        print(f"[WARNING] File {raw_path_types} not found. Skipping FIRE types preparation...\n")
        return

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

    if not raw_path.exists():
        print(f"[WARNING] File {raw_path} not found. Skipping {name.title()} preparation...\n")
        return

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

    if not raw_path.exists():
        print(f"[WARNING] File {raw_path} not found. Skipping {name.title()} preparation...\n")
        return

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