import json
import time
import random

from tqdm import tqdm
from pathlib import Path
from collections import Counter

from src.generators.llm_client import GroqClient
from src.generators.prompt_templates import PromptTemplates
from src.data_loader import load_processed_fire_data


def build_ontology_schema(data, include_types=True, percentage_cutoff=2.0):
    schema_elements = []

    for sample in data:
        for t in sample.triplets:
            if include_types:
                s_type = getattr(t, 'subject_type', 'UNKNOWN_TYPE')
                o_type = getattr(t, 'object_type', 'UNKNOWN_TYPE')
                signature = f"[{s_type}] --({t.relation})--> [{o_type}]"
                schema_elements.append(signature)
            else:
                schema_elements.append(t.relation)

    counts = Counter(schema_elements)
    num_of_elements = len(schema_elements)

    allowed_schemas = []
    schema_weights = []

    # Filter based on the percentage cutoff
    for element, count in counts.items():
        pct = (count / num_of_elements) * 100
        if pct >= percentage_cutoff:
            allowed_schemas.append(element)
            schema_weights.append(pct)

    print(f"Ontology Built: Kept {len(allowed_schemas)} schemas (Cutoff: {percentage_cutoff}%)")
    return allowed_schemas, schema_weights


def sample_schemas(allowed_schemas, schema_weights, n=5):
    sampled = random.choices(
        population=allowed_schemas,
        weights=schema_weights,
        k=n
    )
    return sampled


def main():
    client = GroqClient()

    data_path = Path("data") / "02_processed" / "fire_train_processed.json"

    output_dir = Path("data") / "03_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ontology_2_percent_half_fixed.json"

    data = load_processed_fire_data(data_path)
    allowed_schemas, schema_weights = build_ontology_schema(data, include_types=True, percentage_cutoff=2.0)

    print("\n--------------------")
    print("Starting batch generation")

    num_of_batches = 50
    sentence_per_batch = 10

    system_prompt = PromptTemplates.get_system_prompt_with_template(allowed_schemas)
    #user_prompt = PromptTemplates.generate_n_sentences(sentence_per_batch)

    all_generated_sentences = []

    for i in tqdm(range(num_of_batches)):

        chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=5)
        user_prompt = PromptTemplates.generate_mixed_batch(n=10, forced_schemas=chosen_schemas)

        response = client.generate_message(system_prompt, user_prompt)
        # response = client.print_payload(system_prompt, user_prompt)

        try:
            batch_data = json.loads(response)
            all_generated_sentences.extend(batch_data)
        except json.JSONDecodeError as e:
            print(f"  -> Failed to parse JSON for this batch. Raw output:\n{response}")
        time.sleep(2)

    print(f"\nGeneration complete! Saving {len(all_generated_sentences)} sentences")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_generated_sentences, f, indent=4)

    print(f"Success! Data saved to {output_file}")

if __name__ == "__main__":
    main()