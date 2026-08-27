import json


def find_anomalous_triplets(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    anomalous_sentences = []

    for i, item in enumerate(raw_data):
        triplets = item.get('triplets', [])

        for t in triplets:
            # Check if the rogue 'action' key exists or 'object_type' is missing
            if 'action' in t or 'object_type' not in t:
                batch_id = i // 10  # Calculates which batch this belongs to
                anomalous_sentences.append({
                    "sentence_idx": i,
                    "batch_id": batch_id,
                    "rogue_triplet": t
                })
                break  # Stop checking triplets for this sentence and move to the next

    # Print the findings
    print(f"Found {len(anomalous_sentences)} anomalous sentences.\n")

    for anomaly in anomalous_sentences:
        print(f"Index: {anomaly['sentence_idx']} | Batch: {anomaly['batch_id']}")
        print(f"Triplet: {anomaly['rogue_triplet']}\n")


# Run it on your file
find_anomalous_triplets('data/03_synthetic/entity_orig.json')