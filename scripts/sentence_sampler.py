import json
from pathlib import Path
import random


# Set to None to sample all sentences, OR an integer (e.g., 3) to filter
TARGET_TRIPLET_COUNT = None
NUM_SAMPLES = 5  # Number of sentences to sample per method
random.seed(42)  # Optional: comment out for different random draws

# Setup paths and strategies
fire_path = Path("data/02_processed/fire_sample.json")
llama_dir = Path("data/05_archive/final_llama_temp17/2000_processed")
qwen_dir = Path("data/05_archive/final_qwen_temp17/2000_processed")

strategies = ["zero_shot", "few_shot", "entity", "template"]
strategy_labels = ["Zero-Shot", "Few-Shot", "Entity", "Template"]


# Extract sentences
def load_sentences(file_path, target_count=None):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matching_sentences = []
    for item in data:
        text = item.get("text") or item.get("sentence") or ""
        if not text.strip():
            continue

        # If a target count is provided, strictly filter by it
        if target_count is not None:
            triplets = item.get("triplets") or []
            if len(triplets) != target_count:
                continue

        matching_sentences.append(text.strip())

    return matching_sentences


# Generate dataset lists
datasets_to_sample = [("FIRE Dataset (Baseline)", fire_path)]

for model_name, model_dir in [("Llama", llama_dir), ("Qwen", qwen_dir)]:
    for strat, label in zip(strategies, strategy_labels):
        datasets_to_sample.append(
            (f"{model_name} Model - {label}", model_dir / f"{strat}_2k.json")
        )

# Sample
if TARGET_TRIPLET_COUNT is not None:
    print(f"\n[!] Filtering for sentences with EXACTLY {TARGET_TRIPLET_COUNT} triplet(s)\n")
else:
    print("\n[!] Sampling sentences from ALL triplet counts\n")

for label, path in datasets_to_sample:
    print("=" * 80)
    print(f"METHOD: {label}")
    print(f"Source: {path}")
    print("=" * 80)

    try:
        sentences = load_sentences(path, TARGET_TRIPLET_COUNT)
        total_found = len(sentences)

        if total_found == 0:
            print(f"[!] No sentences found.\n")
            continue

        sample_size = min(NUM_SAMPLES, total_found)
        sampled = random.sample(sentences, sample_size)

        if TARGET_TRIPLET_COUNT is not None:
            print(f"(Found {total_found} matching sentences — displaying {sample_size}):\n")
        else:
            print(f"(Total sentences: {total_found} — displaying {sample_size}):\n")

        for idx, sentence in enumerate(sampled, 1):
            print(f"[{idx}] {sentence}\n")

    except FileNotFoundError:
        print(f"[!] Error: File not found at {path}\n")
    except Exception as e:
        print(f"[!] Error loading {label}: {e}\n")