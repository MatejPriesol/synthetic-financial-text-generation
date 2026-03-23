import json
import time
import random
import argparse

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
    parser = argparse.ArgumentParser(description="Ontology based generation")

    # Path to the data
    parser.add_argument("--test", "-t", action="store_true", help="If enabled, only prints the payload")
    args = parser.parse_args()

    client = GroqClient(temperature=1.3)

    data_path = Path("data") / "02_processed" / "fire_train_processed.json"

    output_dir = Path("data") / "03_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "ontology_2_percent_all_fixed_temp13_schemas_long.json"

    data = load_processed_fire_data(data_path)
    allowed_schemas, schema_weights = build_ontology_schema(data, include_types=True, percentage_cutoff=2.0)

    print("\n--------------------")
    print("Starting batch generation")

    num_of_batches = 250
    sentence_per_batch = 10

    #system_prompt = PromptTemplates.get_system_prompt_with_template(allowed_schemas)
    system_prompt = PromptTemplates.get_system_prompt_with_fixed_templates()
    #user_prompt = PromptTemplates.generate_n_sentences(sentence_per_batch)

    all_generated_sentences = []

    if args.test:
        chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)
        user_prompt = PromptTemplates.generate_batch_from_template(chosen_schemas)
        client.print_payload(system_prompt, user_prompt)
        return

    try:
        for i in tqdm(range(num_of_batches)):

            chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)

            #user_prompt = PromptTemplates.generate_mixed_batch(n=sentence_per_batch, forced_schemas=chosen_schemas)
            user_prompt = PromptTemplates.generate_batch_from_template(chosen_schemas)

            response = client.generate_message(system_prompt, user_prompt)

            # Ran out of the API keys and reached the rate limit, so break
            if response is None:
                print(f"  [!] Used all the rate limits, stopping generation at batch {i+1}")
                break

            try:
                batch_data = json.loads(response)

                # Goal: embed the requested schema for eval purposes
                for j, item in enumerate(batch_data):
                    # Ignore any sentences that go beyond the templates
                    if j >= len(chosen_schemas):
                        break

                    item["requested_schema"] = chosen_schemas[j]
                    all_generated_sentences.append(item)

            except json.JSONDecodeError as e:
                print(f"  -> Failed to parse JSON for this batch. Raw output:\n{response}")

            time.sleep(2)
    except KeyboardInterrupt:
        print("\n[!] Script manually stopped.")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}.")
    finally:
        print(f"\nGeneration stopped/finished. Saving {len(all_generated_sentences)} sentences...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_generated_sentences, f, indent=4)
        print(f"Success! Data saved to {output_file}")

if __name__ == "__main__":
    main()