import json


def remove_batch(input_filepath, output_filepath, batch_id, batch_size=10):
    # Load the original data
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    start_idx = batch_id * batch_size
    end_idx = start_idx + batch_size

    print(f"Original dataset size: {len(data)}")

    # Check if the batch ID is valid
    if start_idx >= len(data):
        print("Error: Batch ID is out of range!")
        return

    print(f"Removing batch {batch_id} (indices {start_idx} to {end_idx - 1})...")

    # Slice the list: Keep everything BEFORE the batch, and everything AFTER the batch
    cleaned_data = data[:start_idx] + data[end_idx:]

    print(f"Cleaned dataset size: {len(cleaned_data)}")

    # Save to a new file, keeping the JSON structure and indentation pretty
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    print(f"Saved successfully to {output_filepath}")


# --- RUN IT ---
# Replace with your actual file names
INPUT_FILE = "data/03_synthetic/entity_orig.json"
OUTPUT_FILE = "data/03_synthetic/entity.json"
BATCH_TO_REMOVE = 13

remove_batch(INPUT_FILE, OUTPUT_FILE, BATCH_TO_REMOVE)