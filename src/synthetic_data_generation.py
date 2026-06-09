import json
import time
import random
import argparse
import concurrent.futures

from tqdm import tqdm
from pathlib import Path
from collections import Counter

from src.generators.llm_client import GroqClient, LocalClient
from src.generators.prompt_templates import PromptTemplates
from src.data_loader import load_processed_data

from collections import defaultdict, Counter


def build_template_sets_from_data(data, include_types=True, max_triplets=5):
    """
    Builds coherent schema templates by grouping triplets exactly as they
    co-occurred in the original dataset sentences.
    """
    # Dictionary to hold Counters for each length: {1: Counter(), 2: Counter()...}
    length_counters = defaultdict(Counter)

    for sample in data:
        signatures = []

        # Build the individual triplet signatures for this sentence
        for t in sample.triplets:
            if include_types:
                s_type = getattr(t, 'subject_type', 'UNKNOWN_TYPE')
                o_type = getattr(t, 'object_type', 'UNKNOWN_TYPE')
                sig = f"[{s_type}] --({t.relation})--> [{o_type}]"
                signatures.append(sig)
            else:
                signatures.append(t.relation)

        length = len(signatures)

        # Only process sentences that have between 1 and max_triplets
        if 1 <= length <= max_triplets:
            # Sort them so (A, B) is counted as the same template as (B, A)
            sorted_sigs = tuple(sorted(signatures))
            length_counters[length][sorted_sigs] += 1

    # Convert the raw counts into usable arrays with percentage weights
    templates_by_length = {}

    for length, counter in length_counters.items():
        total_occurrences = sum(counter.values())

        templates = []
        weights = []

        for template_tuple, count in counter.items():
            templates.append(template_tuple)
            # Calculate the probability weight (0.0 to 1.0)
            weights.append(count / total_occurrences)

        templates_by_length[length] = {
            "templates": templates,
            "weights": weights
        }

        print(f"Length {length}: Found {len(templates)} unique templates from {total_occurrences} sentences.")

    return templates_by_length


def build_system_prompt_schema(data, include_types=True, percentage_cutoff=2.0):
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

    # Calculate weights
    counts = Counter(schema_elements)
    num_of_elements = len(schema_elements)

    allowed_schemas = []
    schema_weights = []

    for element, count in counts.items():
        pct = (count / num_of_elements) * 100
        if pct >= percentage_cutoff:
            allowed_schemas.append(element)
            schema_weights.append(pct)

    print(f"System Prompt Schema Built: Kept {len(allowed_schemas)} unique signatures (Cutoff: {percentage_cutoff}%)")
    return allowed_schemas, schema_weights


def build_entity_schema(data, allowed_type_schemas, frequency_threshold=1):
    """ Builds schemas based on literal entity names, strictly filtered by the allowed type-level ontology."""
    schema_elements = []

    for sample in data:
        for t in sample.triplets:
            # Reconstruct the Type-Level signature to check if it's allowed
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')
            type_signature = f"[{s_type}] --({t.relation})--> [{o_type}]"

            # If the schema is in our allowed schemas, extract the literal entities
            if type_signature in allowed_type_schemas:
                subject = getattr(t, 'subject', 'UNKNOWN')
                obj = getattr(t, 'object', 'UNKNOWN')

                # Generate the entity signature that will be passed into the LLM
                entity_signature = f"[{subject}] --({t.relation})--> [{obj}] (Ontology Schema: [{s_type}] --({t.relation})--> [{o_type}])"
                schema_elements.append((entity_signature, type_signature))

    # Calculate weights
    counts = Counter(schema_elements)

    allowed_schemas = []
    schema_weights = []

    for element, count in counts.items():
        if count >= frequency_threshold:
            allowed_schemas.append(element)
            schema_weights.append(count)

    print(f"Entity Schema Built: Kept {len(allowed_schemas)} unique literal signatures.")
    return allowed_schemas, schema_weights

def sample_schemas(allowed_schemas, schema_weights, n=10):
    sampled = random.choices(
        population=allowed_schemas,
        weights=schema_weights,
        k=n
    )
    return sampled


def get_valid_few_shot_pool(data, allowed_type_schemas):
    """Filters the dataset to only include sentences where ALL triplets are allowed."""
    valid_samples = []

    for sample in data:
        is_valid = True
        for t in sample.triplets:
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')
            type_signature = f"[{s_type}] --({t.relation})--> [{o_type}]"

            if type_signature not in allowed_type_schemas:
                is_valid = False
                break

        # Only keep it if it's completely valid and actually has triplets
        if is_valid and len(sample.triplets) > 0:
            valid_samples.append(sample)

    print(f"Few-Shot Pool Built: Kept {len(valid_samples)} strictly valid sentences out of {len(data)}.")
    return valid_samples


def sample_few_shot_examples(valid_pool, n=3):
    """Samples valid sentences and formats them exactly how the LLM should output them."""
    n = min(n, len(valid_pool))
    sampled_objects = random.sample(valid_pool, k=n)

    formatted_examples = []
    for sample in sampled_objects:
        # Build the exact dictionary structure expected by the system prompt
        example_dict = {
            "text": getattr(sample, 'sentence', ''),
            "triplets": []
        }

        for t in sample.triplets:
            example_dict["triplets"].append({
                "subject": getattr(t, 'subject', ''),
                "relation": getattr(t, 'relation', ''),
                "object": getattr(t, 'object', ''),
                "subject_type": getattr(t, 'subject_type', 'UNKNOWN'),
                "object_type": getattr(t, 'object_type', 'UNKNOWN')
            })

        # Convert the dictionary to JSON string
        formatted_examples.append(json.dumps(example_dict, indent=2))

    return formatted_examples

def sample_sentences(data, n=3):
    n = min(n, len(data))
    sampled_objects = random.sample(data, k=n)
    return [getattr(item, 'sentence', str(item)) for item in sampled_objects]

def sample_grouped_schemas(grouped_templates, n=10):
    """
    Samples a batch of template sets using the true length distribution
    from the FIRE dataset.
    """
    # The true distribution of lengths from the FIRE dataset
    length_options = [1, 2, 3, 4, 5]
    length_weights = [30.5, 26.8, 20.2, 14.6, 8.0]

    # 1. Decide the architecture (number of triplets) for the n sentences
    chosen_lengths = random.choices(length_options, weights=length_weights, k=n)

    sampled_tuples = []

    # 2. Pick the specific grouped templates for those lengths
    for target_length in chosen_lengths:
        pool = grouped_templates.get(target_length)

        # Fallback just in case a length is missing (though our dict has 1-5)
        if not pool:
            continue

        # Pick ONE template tuple based on the real co-occurrence weights
        chosen_template_tuple = random.choices(pool["templates"], weights=pool["weights"], k=1)[0]
        sampled_tuples.append(chosen_template_tuple)

    return sampled_tuples

def prepare_batch_prompts(strategy, valid_pool, allowed_schemas, schema_weights, sentence_per_batch, grouped_templates):
    chosen_schemas = []
    system_prompt = ""
    user_prompt = ""

    if strategy == "template":
        #chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)
        # user_prompt = PromptTemplates.get_user_prompt_template(chosen_schemas)
        system_prompt = PromptTemplates.get_system_prompt_template()
        chosen_schemas = sample_grouped_schemas(grouped_templates, n=sentence_per_batch)
        user_prompt = PromptTemplates.get_user_prompt_template_new(chosen_schemas)

    elif strategy == "entity":
        chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)
        # In case of entity strategy, the schemas are tuples containing the prompt string and evaluation string, extract the prompt
        system_prompt = PromptTemplates.get_system_prompt_entity()
        prompt_strings = [schema[0] for schema in chosen_schemas]
        user_prompt = PromptTemplates.get_user_prompt_entity(prompt_strings)

    elif strategy == "few-shot":
        few_shot_examples = sample_few_shot_examples(valid_pool, n=3)
        system_prompt = PromptTemplates.get_system_prompt_in_context(allowed_schemas)
        user_prompt = PromptTemplates.get_user_prompt_few_shot(sentence_per_batch, few_shot_examples)

    elif strategy == "zero-shot":
        system_prompt = PromptTemplates.get_system_prompt_in_context(allowed_schemas)
        user_prompt = PromptTemplates.get_user_prompt_zero_shot(sentence_per_batch)

    return system_prompt, user_prompt, chosen_schemas

def main():
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")

    parser.add_argument("--test", "-t", action="store_true", help="If enabled, only prints the payload without generating")
    parser.add_argument("--strategy", type=str, choices=["template", "entity", "few-shot", "zero-shot"], required=True, help="Which generation strategy to use")
    parser.add_argument("--cutoff", type=float, default=None, help="Percentage cutoff for sampling")
    parser.add_argument("--output", "-o", type=str, help="Name for the json output file")

    parser.add_argument("--provider", type=str, choices=["api", "local"], default="local", help="Which LLM provider to use")

    args = parser.parse_args()

    # Setup
    if args.provider == "api":
        print("Using API provider")
        client = GroqClient(temperature=1.3)
    else:
        print("Using local provider")
        client = LocalClient(temperature=1.3)
    data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    data = load_processed_data(data_path)

    output_dir = Path("data") / "03_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the system prompt with the list of most used triplets
    cutoff = args.cutoff if args.cutoff is not None else 2.0
    allowed_schemas, schema_weights = build_system_prompt_schema(data, include_types=True, percentage_cutoff=cutoff)
    #system_prompt = PromptTemplates.get_system_prompt_template()
    file_name = args.output if args.output else f"{args.strategy}_synthetic_data.json"
    output_file = output_dir / file_name

    schema_path = output_dir / f"allowed_schemas_cutoff_{cutoff}%.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(allowed_schemas, f, indent=4)

    # In case I am using the entity strategy, I need to pass the literal entities into the user prompt, so have to update the schemas
    if args.strategy == "entity":
        allowed_schemas, schema_weights = build_entity_schema(data, allowed_schemas, frequency_threshold=1)

    valid_pool = data
    if args.strategy == "few-shot":
        valid_pool = get_valid_few_shot_pool(data, allowed_schemas)


    print("\n--------------------")
    print("Starting batch generation")

    num_of_batches = 10
    sentence_per_batch = 10
    all_generated_sentences = []

    templates_by_length = build_template_sets_from_data(data)

    if args.test:

        system_prompt, user_prompt, chosen_schemas = prepare_batch_prompts(
            args.strategy, valid_pool, allowed_schemas, schema_weights, sentence_per_batch, templates_by_length
        )

        client.print_payload(system_prompt, user_prompt)
        return

    def process_batch():
        # Prepare the prompt
        system_prompt, user_prompt, chosen_schemas = prepare_batch_prompts(
            args.strategy, valid_pool, allowed_schemas, schema_weights, sentence_per_batch, templates_by_length
        )

        # Send payload to the model
        response = client.generate_message(system_prompt, user_prompt)

        if response is None:
            return []  # Return empty list if this specific call fails

        batch_results = []
        try:
            batch_data = json.loads(response)

            # Assign schemas for evaluation based on strategy
            for j, item in enumerate(batch_data):
                if args.strategy == "template":
                    # Ignore any sentences that go beyond the templates
                    if j >= len(chosen_schemas):
                        break
                    item["requested_schema"] = chosen_schemas[j]
                # In case of entity strategy, the chosen schemas is a tuple containing the prompt triplet and the eval triplet
                elif args.strategy == "entity":
                    if j >= len(chosen_schemas):
                        break
                    item["requested_schema"] = chosen_schemas[j][1]
                    item["requested_entity_fact"] = chosen_schemas[j][0]
                else:
                    # Set to None so the template adherence won't run
                    item["requested_schema"] = None

                batch_results.append(item)

        except json.JSONDecodeError:
            pass

        return batch_results

    max_workers = 10 if args.provider == "local" else 2

    try:
        # Launch the ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches to the executor at once
            futures = [executor.submit(process_batch) for _ in range(num_of_batches)]

            # Wrap in tqdm to update the progress bar as they finish (not necessarily in order)
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_of_batches):
                try:
                    result = future.result()
                    if result:
                        all_generated_sentences.extend(result)
                except Exception as exc:
                    print(f"  [!] A batch generated an exception: {exc}")

                if args.provider == "api":
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