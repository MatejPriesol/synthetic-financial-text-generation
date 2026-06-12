import json
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Prune generated dataset to exactly 2,000 entries (unless different size is specified).")
    parser.add_argument("--input", type=str, required=True, help="Path to raw generated JSON")
    parser.add_argument("--output", type=str, required=True, help="Path to save the pruned 2K JSON")
    parser.add_argument("--target_size", type=int, default=2000, help="Target dataset size")
    args = parser.parse_args()

    # Load raw data
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    current_size = len(data)
    print(f"Loaded dataset with {current_size} entries.")

    if current_size < args.target_size:
        raise ValueError(f"Dataset only has {current_size} entries. Cannot prune to {args.target_size}!")

    random.seed(42)

    pruned_data = random.sample(data, args.target_size)

    # Save the pruned dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pruned_data, f, indent=4)

    print(f"Successfully created pruned dataset at {args.output} with exactly {len(pruned_data)} entries.")


if __name__ == "__main__":
    main()