import json
import os
from dataclasses import asdict
from src.data_loader import load_fire_data, load_fire_relations


def main():
    raw_path_data = "data/01_raw/fire_train.json"
    raw_path_relations = "data/01_raw/fire_types.json"

    processed_dir = "data/02_processed"
    os.makedirs(processed_dir, exist_ok=True)
    output_path_data = os.path.join(processed_dir, "fire_train_processed.json")
    output_path_relations = os.path.join(processed_dir, "fire_types_processed.json")

    print("Loading and parsing raw FIRE data")
    processed_data = load_fire_data(raw_path_data)

    # Convert to dict
    data_to_save = [asdict(sample) for sample in processed_data]

    # Save into file
    with open(output_path_data, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4)

    print(f"Data saved to {output_path_data}")


    print("Loading and parsing FIRE relations")
    relations = load_fire_relations(raw_path_relations)

    with open(output_path_relations, 'w', encoding='utf-8') as f:
        json.dump(relations, f, indent=4)
    print(f"Relations saved to {output_path_relations}")


if __name__ == "__main__":
    main()