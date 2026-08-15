import json
import time
import random
import math
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


def build_global_entity_triplet_pool(data, allowed_type_schemas, pool_size=500):
    """
    Extracts literal triplet facts that conform to the allowed type ontology,
    counts their frequencies, and returns the top 'pool_size' elements with weights.
    """
    literal_triplets = []

    for sample in data:
        for t in sample.triplets:
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')
            type_signature = f"[{s_type}] --({t.relation})--> [{o_type}]"

            # Only track if it matches our 16 allowed ontological paths
            if type_signature in allowed_type_schemas:
                subject = getattr(t, 'subject', 'UNKNOWN')
                obj = getattr(t, 'object', 'UNKNOWN')

                # Save the complete literal fact with its types
                literal_fact = (subject, t.relation, obj, s_type, o_type)
                literal_triplets.append(literal_fact)

    # Get the top X most frequent literal facts
    counts = Counter(literal_triplets)
    top_elements = counts.most_common(pool_size)

    global_pool = [element for element, count in top_elements]
    # Calculate weights based on frequency counts
    total_counts = sum(count for element, count in top_elements)
    global_weights = [count / total_counts for element, count in top_elements]

    print(f"Entity Strategy: Built a global pool of {len(global_pool)} unique literal facts.")
    return global_pool, global_weights

def sample_and_format_entity_pool(entity_pool, entity_weights, n_facts_to_sample=20):
    """
    Samples a set of connected literal facts from the global pool,
    strips the relations, and groups unique entities by their types
    into a clean string for the LLM user prompt.
    """
    # Sample literal facts based on their frequency weights
    sampled_facts = random.choices(entity_pool, weights=entity_weights, k=n_facts_to_sample)

    # Group unique literal entities by their types
    type_to_entities = defaultdict(set)
    for subject, relation, obj, s_type, o_type in sampled_facts:
        type_to_entities[s_type].add(subject)
        type_to_entities[o_type].add(obj)

    # Format into a clean string for the prompt
    formatted_lines = []
    requested_entity_pool = []
    for entity_type, entities in type_to_entities.items():
        # Convert set to a sorted list for consistent prompting
        entity_list = sorted(list(entities))
        requested_entity_pool.extend(entity_list)
        formatted_lines.append(f"- {entity_type}: {entity_list}")

    # Return both the prompt string and the underlying facts (for downstream evaluation)
    prompt_string = "\n".join(formatted_lines)
    return prompt_string, requested_entity_pool


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

        # Only keep it if it's completely valid and has between 1 and 5 triplets included
        if is_valid and 0 < len(sample.triplets) <= 5:
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

def save_template_schemas(templates_by_length, output_path):
    """
    Flattens templates of all lengths into a unique list of individual
    triplet signatures and saves them to a JSON file for compliance checking.
    """
    flattened_triplets = set()

    for length, data in templates_by_length.items():
        for template_tuple in data["templates"]:
            for triplet_str in template_tuple:
                flattened_triplets.add(triplet_str)

    allowed_template_ontology = sorted(list(flattened_triplets))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(allowed_template_ontology, f, indent=4)

    #print(f"\nSaved {len(allowed_template_ontology)} unique template triplets to {output_path}")

def get_density_blueprint(batch_size=10):
    triplet_targets = [1, 2, 3, 4, 5]
    # The true distribution of lengths from the FIRE dataset
    length_weights = [30.5, 26.8, 20.2, 14.6, 8.0]

    # Probabilistic sampling based on the FIRE distribution
    sampled_lengths = random.choices(
        population=triplet_targets,
        weights=length_weights,
        k=batch_size
    )

    # Format for the prompt
    formatted_lines = []
    for i, target in enumerate(sampled_lengths, 1):
        formatted_lines.append(f"- Sentence {i}: Must contain exactly {target} triplet(s).")

    return "\n".join(formatted_lines), sampled_lengths

def prepare_batch_prompts(strategy, valid_pool, allowed_ontology, schema_weights, sentence_per_batch, grouped_templates, entity_pool, entity_weights):
    chosen_schemas = []
    blueprint_counts = []
    system_prompt = ""
    user_prompt = ""

    if strategy == "template":
        #chosen_schemas = sample_schemas(allowed_schemas, schema_weights, n=sentence_per_batch)
        # user_prompt = PromptTemplates.get_user_prompt_template(chosen_schemas)
        system_prompt = PromptTemplates.get_system_prompt_template()
        chosen_schemas = sample_grouped_schemas(grouped_templates, n=sentence_per_batch)
        user_prompt = PromptTemplates.get_user_prompt_template_new(chosen_schemas)
        blueprint_counts = [len(schema_tuple) for schema_tuple in chosen_schemas]

    elif strategy == "entity":
        blueprint_string, blueprint_counts = get_density_blueprint(batch_size=sentence_per_batch)
        system_prompt = PromptTemplates.get_system_prompt_entity(allowed_ontology)
        pool_string, underlying_facts = sample_and_format_entity_pool(entity_pool, entity_weights, n_facts_to_sample=20)
        user_prompt = PromptTemplates.get_user_prompt_entity(pool_string, blueprint_string)
        chosen_schemas = underlying_facts

    elif strategy == "few-shot":
        blueprint_string, blueprint_counts = get_density_blueprint(batch_size=sentence_per_batch)
        few_shot_examples = sample_few_shot_examples(valid_pool, n=3)
        system_prompt = PromptTemplates.get_system_prompt_in_context(allowed_ontology)
        user_prompt = PromptTemplates.get_user_prompt_few_shot(sentence_per_batch, few_shot_examples, blueprint_string)

    elif strategy == "zero-shot":
        blueprint_string, blueprint_counts = get_density_blueprint(batch_size=sentence_per_batch)
        system_prompt = PromptTemplates.get_system_prompt_in_context(allowed_ontology)
        user_prompt = PromptTemplates.get_user_prompt_zero_shot(sentence_per_batch, blueprint_string)

    return system_prompt, user_prompt, chosen_schemas, blueprint_counts

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
        client = LocalClient(temperature=1.7)
    data_path = Path("data") / "02_processed" / "fire_all_processed.json"
    data = load_processed_data(data_path)

    output_dir = Path("data") / "03_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = args.output if args.output else f"{args.strategy}_synthetic_data.json"
    output_file = output_dir / file_name

    # Get the system prompt with the list of most used triplets
    cutoff = args.cutoff if args.cutoff is not None else 2.0
    allowed_ontology, schema_weights = build_system_prompt_schema(data, include_types=True, percentage_cutoff=cutoff)

    schema_path = output_dir / f"allowed_schemas_cutoff_{cutoff}%.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(allowed_ontology, f, indent=4)

    entity_pool, entity_weights = [], []
    if args.strategy == "entity":
        entity_pool, entity_weights = build_global_entity_triplet_pool(data, allowed_ontology, pool_size=500)

    valid_pool = data
    if args.strategy == "few-shot":
        valid_pool = get_valid_few_shot_pool(data, allowed_ontology)


    print("\n--------------------")
    print("Starting batch generation")

    # Config
    target_dataset_size = 2000
    sentence_per_batch = 10
    safety_buffer_percentage = 10

    total_target_sentences = target_dataset_size * (1 + safety_buffer_percentage / 100)
    num_of_batches = math.ceil(total_target_sentences / sentence_per_batch)

    all_generated_sentences = []

    templates_by_length = build_template_sets_from_data(data)

    template_json_path = output_dir / "allowed_schemas_template.json"

    # Extract and save them
    save_template_schemas(templates_by_length, template_json_path)

    if args.test:

        system_prompt, user_prompt, chosen_schemas, blueprint_counts = prepare_batch_prompts(
            args.strategy, valid_pool, allowed_ontology, schema_weights, sentence_per_batch, templates_by_length, entity_pool, entity_weights
        )

        client.print_payload(system_prompt, user_prompt)
        return

    def process_batch(i):
        # Prepare the prompt
        system_prompt, user_prompt, chosen_schemas, blueprint_counts = prepare_batch_prompts(
            args.strategy, valid_pool, allowed_ontology, schema_weights, sentence_per_batch, templates_by_length, entity_pool, entity_weights
        )

        # Send payload to the model
        response = client.generate_message(system_prompt, user_prompt)

        if response is None:
            return [], "api_failure"  # Return empty list if this specific call fails

        batch_results = []
        status = "exact_match"

        try:
            batch_data = json.loads(response)

            if isinstance(batch_data, dict):
                batch_data = [batch_data]

            # Determine if the generated count of sentences was good
            generated_count = len(batch_data)
            requested_count = len(blueprint_counts)

            if generated_count > requested_count:
                status = "over_generated"
                return [], status
            elif generated_count < requested_count:
                status = "under_generated"
                return [], status

            # Assign schemas for evaluation based on strategy
            for j, item in enumerate(batch_data):

                # If the model generates more sentences than requested, break
                if j >= len(blueprint_counts):
                    break

                item["requested_triplet_count"] = blueprint_counts[j]  # Attach the requested triplet count

                if args.strategy == "template":
                    item["requested_schema"] = chosen_schemas[j]
                # In case of entity strategy, the chosen schemas is a tuple containing the prompt triplet and the eval triplet
                elif args.strategy == "entity":
                    item["requested_schema"] = chosen_schemas
                else:
                    # Set to None so the template adherence won't run
                    item["requested_schema"] = None

                batch_results.append(item)

        except json.JSONDecodeError:
            status = "json_failure"

        return batch_results, status

    max_workers = 10 if args.provider == "local" else 2
    run_stats = Counter()

    try:
        # Launch the ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batches to the executor at once
            futures = [executor.submit(process_batch, i) for i in range(num_of_batches)]

            # Wrap in tqdm to update the progress bar as they finish (not necessarily in order)
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_of_batches):
                try:
                    result, status = future.result()
                    run_stats[status] += 1

                    if result:
                        all_generated_sentences.extend(result)
                except Exception as exc:
                    run_stats["code_exception"] += 1
                    print(f"  [!] A batch generated an exception: {exc}")

                #if args.provider == "api":
                    #time.sleep(2)

    except KeyboardInterrupt:
        print("\n[!] Script manually stopped.")
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}.")
    finally:
        print("\n" + "=" * 40)
        print("FINAL BATCH STATISTICS:")
        for status_type, count in run_stats.items():
            print(f" - {status_type}: {count}")
        print("=" * 40)

        print(f"\nGeneration stopped/finished. Saving {len(all_generated_sentences)} sentences...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_generated_sentences, f, indent=4)
        print(f"Success! Data saved to {output_file}")

if __name__ == "__main__":
    main()