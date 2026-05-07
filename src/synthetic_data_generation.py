import json
import time
import random
import argparse

from tqdm import tqdm
from pathlib import Path
from collections import Counter

from src.generators.llm_client import GroqClient
from src.generators.prompt_templates import PromptTemplates
from src.data_loader import load_processed_data


def build_template_schema(data, include_types=True, percentage_cutoff=2.0):
    """Builds schemas based on semantic types."""
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

    return _calculate_weights(schema_elements, percentage_cutoff, "Ontology Schema")

def build_entity_schema(data, percentage_cutoff=0.1):
    """Builds schemas based on literal entity names."""
    schema_elements = []
    for sample in data:
        for t in sample.triplets:
            # We use the actual text of the subject and object here
            subject = getattr(t, 'subject', 'UNKNOWN')
            obj = getattr(t, 'object', 'UNKNOWN')
            signature = f"[{subject}] --({t.relation})--> [{obj}]"
            schema_elements.append(signature)

    return _calculate_weights(schema_elements, percentage_cutoff, "Entity Schema")

def _calculate_weights(schema_elements, cutoff, name="Schema"):
    """Helper function to calculate weights and apply cutoffs."""
    counts = Counter(schema_elements)
    num_of_elements = len(schema_elements)

    allowed_schemas = []
    schema_weights = []

    for element, count in counts.items():
        pct = (count / num_of_elements) * 100
        if pct >= cutoff:
            allowed_schemas.append(element)
            schema_weights.append(pct)

    print(f"{name} Built: Kept {len(allowed_schemas)} unique signatures (Cutoff: {cutoff}%)")
    return allowed_schemas, schema_weights

def sample_schemas(allowed_schemas, schema_weights, n=10):
    sampled = random.choices(
        population=allowed_schemas,
        weights=schema_weights,
        k=n
    )
    return sampled

def sample_sentences(data, n=3):
    n = min(n, len(data))
    sampled_objects = random.sample(data, k=n)
    return [getattr(item, 'sentence', str(item)) for item in sampled_objects]

def prepare_batch_prompts(strategy, data, allowed_schemas, schema_weights, sentence_per_batch):
    chosen_schemas = []
    user_prompt = ""

    if strategy == "template":
        chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)
        user_prompt = PromptTemplates.get_user_prompt_template(chosen_schemas)

    elif strategy == "entity":
        # TODO: Maybe change required with the sampling
        chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)
        user_prompt = PromptTemplates.get_user_prompt_entity(chosen_schemas)

    elif strategy == "few-shot":
        few_shot_examples = sample_sentences(data, n=3)
        user_prompt = PromptTemplates.get_user_prompt_few_shot(sentence_per_batch, few_shot_examples)

    elif strategy == "zero-shot":
        user_prompt = PromptTemplates.get_user_prompt_zero_shot(sentence_per_batch)

    return user_prompt, chosen_schemas

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")

    parser.add_argument("--test", "-t", action="store_true", help="If enabled, only prints the payload without generating")
    parser.add_argument("--strategy", type=str, choices=["template", "entity", "few-shot", "zero-shot"], required=True, help="Which generation strategy to use")
    parser.add_argument("--cutoff", type=float, default=None, help="Percentage cutoff for sampling")
    parser.add_argument("--output", "-o", type=str, help="Name for the json output file")

    args = parser.parse_args()

    # Setup
    client = GroqClient(temperature=1.3)
    data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    data = load_processed_data(data_path)

    output_dir = Path("data") / "03_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_schemas, schema_weights, file_name, system_prompt = [], [], "", ""

    # Finish setup relevant to strategy
    if args.strategy == "template":
        cutoff = args.cutoff if args.cutoff is not None else 2.0
        allowed_schemas, schema_weights = build_template_schema(data, include_types=True, percentage_cutoff=cutoff)
        #system_prompt = PromptTemplates.get_system_prompt_with_fixed_templates()
        system_prompt = PromptTemplates.get_system_prompt_universal(allowed_schemas)
        file_name = args.output if args.output else "ontology_synthetic_data.json"

    elif args.strategy == "entity":
        cutoff = args.cutoff if args.cutoff is not None else 0.1
        allowed_schemas, schema_weights = build_entity_schema(data, percentage_cutoff=cutoff)
        # TODO: Change so it uses the same allowed schemas, the sampling later will be from literal entities.
        system_prompt = PromptTemplates.get_system_prompt_universal(allowed_schemas)
        file_name = args.output if args.output else "entity_synthetic_data.json"

    elif args.strategy == "few-shot":
        cutoff = args.cutoff if args.cutoff is not None else 2.0
        allowed_schemas, schema_weights = build_template_schema(data, include_types=True, percentage_cutoff=cutoff)
        # system_prompt = PromptTemplates.get_system_prompt_with_fixed_templates()
        system_prompt = PromptTemplates.get_system_prompt_universal(allowed_schemas)
        file_name = args.output if args.output else "few-shot_synthetic_data.json"

    elif args.strategy == "zero-shot":
        cutoff = args.cutoff if args.cutoff is not None else 2.0
        allowed_schemas, schema_weights = build_template_schema(data, include_types=True, percentage_cutoff=cutoff)
        # system_prompt = PromptTemplates.get_system_prompt_with_fixed_templates()
        system_prompt = PromptTemplates.get_system_prompt_universal(allowed_schemas)
        file_name = args.output if args.output else "zero-shot_synthetic_data.json"

    output_file = output_dir / file_name

    print("\n--------------------")
    print("Starting batch generation")

    num_of_batches = 10
    sentence_per_batch = 10
    all_generated_sentences = []

    if args.test:
        user_prompt, chosen_schemas = prepare_batch_prompts(
            args.strategy, data, allowed_schemas, schema_weights, sentence_per_batch
        )

        client.print_payload(system_prompt, user_prompt)
        return

    try:
        for i in tqdm(range(num_of_batches)):

            user_prompt, chosen_schemas = prepare_batch_prompts(
                args.strategy, data, allowed_schemas, schema_weights, sentence_per_batch
            )

            # Generate the message
            response = client.generate_message(system_prompt, user_prompt)

            # Ran out of the API keys and reached the rate limit, so break
            if response is None:
                print(f"  [!] Used all the rate limits, stopping generation at batch {i+1}")
                break

            try:
                batch_data = json.loads(response)

                # Goal: embed the requested schema for eval purposes, relevant for template and entity
                for j, item in enumerate(batch_data):
                    if args.strategy in ["template", "entity"]:
                        # Ignore any sentences that go beyond the templates
                        if j >= len(chosen_schemas):
                            break
                        item["requested_schema"] = chosen_schemas[j]
                    else:
                        # Set to None so the template adherence won't run
                        item["requested_schema"] = None
                    all_generated_sentences.append(item)

            except json.JSONDecodeError as e:
                print(f"  -> Failed to parse JSON for batch {i+1}")

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