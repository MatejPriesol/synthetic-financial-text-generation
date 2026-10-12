import json
import re
from pathlib import Path

# Data paths
fire_path = Path("data/02_processed/fire_sample.json")
llama_dir = Path("data/05_archive/final_llama_temp17/2000_processed")
qwen_dir = Path("data/05_archive/final_qwen_temp17/2000_processed")
files = ["zero_shot_2k.json", "few_shot_2k.json", "template_2k.json", "entity_2k.json"]

phrases_to_search = ["financial entity", "action", "productof", "Actionto", "Valuein", "designation"]


# For a given dataset, count each phrase across all sentences
def count_phrases_in_dataset(filepath, phrases):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return {p: "N/A" for p in phrases}

    # Compile regex patterns for exact word boundaries, case-insensitive
    patterns = {p: re.compile(rf"\b{p}\b", re.IGNORECASE) for p in phrases}
    counts = {p: 0 for p in phrases}

    for item in data:
        # FIRE uses "sentence", synthetic uses "text"
        text = item.get("text") or item.get("sentence") or ""

        for phrase, pattern in patterns.items():
            if pattern.search(text):
                counts[phrase] += 1

    return counts


results = {}

# Evaluate FIRE
results["FIRE Baseline"] = count_phrases_in_dataset(fire_path, phrases_to_search)

# Evaluate LLMs
for file in files:
    strat_name = file.replace("_2k.json", "").replace("_", " ").title()

    results[f"Llama {strat_name}"] = count_phrases_in_dataset(llama_dir / file, phrases_to_search)
    results[f"Qwen {strat_name}"] = count_phrases_in_dataset(qwen_dir / file, phrases_to_search)


# Print results
dataset_col_width = max(len("Dataset"), max(len(name) for name in results.keys()))

col_widths = {}
for phrase in phrases_to_search:
    # Find the maximum string length of the values (or default to the header's length)
    max_val_len = max((len(str(counts.get(phrase, "N/A"))) for counts in results.values()), default=0)
    col_widths[phrase] = max(len(phrase.title()), max_val_len)

# Build and print the header row
header_parts = [f"{'Dataset'.ljust(dataset_col_width)}"]
for phrase in phrases_to_search:
    header_parts.append(f"{phrase.title().ljust(col_widths[phrase])}")
header_str = " | ".join(header_parts)

print(header_str)
print("-" * len(header_str))

# Build and print the data rows
for name, counts in results.items():
    row_parts = [f"{name.ljust(dataset_col_width)}"]
    for phrase in phrases_to_search:
        val_str = str(counts.get(phrase, "N/A"))
        row_parts.append(f"{val_str.ljust(col_widths[phrase])}")
    print(" | ".join(row_parts))