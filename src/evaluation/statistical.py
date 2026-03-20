import argparse
import json

from collections import Counter
from src.data_loader import load_data_for_eval

def evaluate_obj_subj(synthetic_data, top_n=10, verbose=False):
    subjects = [triplet.subject for sample in synthetic_data for triplet in sample.triplets]
    objects = [triplet.object for sample in synthetic_data for triplet in sample.triplets]

    subject_counts = Counter(subjects)
    object_counts = Counter(objects)

    top_subjects = subject_counts.most_common(top_n)
    top_objects = object_counts.most_common(top_n)

    if verbose:
        print(f"{'Top Subjects (Heads)':<25} | {'Top Objects (Tails)'}")
        print("-" * 50)

        for i in range(top_n):
            s_str = f"{top_subjects[i][0]} ({top_subjects[i][1]})" if i < len(top_subjects) else ""
            o_str = f"{top_objects[i][0]} ({top_objects[i][1]})" if i < len(top_objects) else ""

            print(f"{s_str:<25} | {o_str}")

    return {
        "top_subjects": {k: v for k, v in top_subjects},
        "top_objects": {k: v for k, v in top_objects}
    }


def profile_ontology(data, include_types=False, top_n=10, verbose=False):
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

    if verbose:
        print(f"\n{'Count':>7} | {'Triplet (or relation if no entity types)'}")
        print("-" * 60)

        for element, count in counts.most_common(top_n):
            pct_str = f"{(count / num_of_elements) * 100:.2f}%"
            print(f"{pct_str:>7} | {element}")

    return {
        "total_triplets_extracted": num_of_elements,
        "unique_schemas_used": len(counts),
        "top_schemas": {element: count for element, count in counts.most_common(top_n)}
    }


def evaluate_template_adherence(data, verbose=False):
    matched_count = 0
    total_valid_samples = 0

    if verbose:
        print("\n Template Adherence Mismatches:")

    for sample in data:
        req_schema = getattr(sample, 'requested_schema', None)

        # If this dataset wasn't generated with templates, skip it
        if not req_schema:
            continue

        total_valid_samples += 1

        # Reconstruct the generated schemas from the sample's triplets
        gen_schemas = []
        for t in sample.triplets:
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')
            gen_schemas.append(f"[{s_type}] --({t.relation})--> [{o_type}]")

        # Check if at least one of the triplets followed the schema
        if req_schema in gen_schemas:
            matched_count += 1
        else:
            if verbose:
                print(f"FAILED:\n  Sentence:  {sample.sentence}")
                print(f"  Requested: {req_schema}")
                print(f"  Generated: {gen_schemas}\n")

    # Calculate the final score
    adherence_score = (matched_count / total_valid_samples) * 100 if total_valid_samples > 0 else 0

    if verbose and 0 < total_valid_samples == matched_count:
        print("100% Adherence, no mismatches found.\n")

    return {
        "adherence_percentage": round(adherence_score, 2),
        "total_checked": total_valid_samples,
        "total_matched": matched_count
    }

# Check if it contains the types (under the assumption that if the first one does, they all do) for the ontology
def has_types_helper(data):
    if len(data) > 0 and len(data[0].triplets) > 0:
        first_triplet = data[0].triplets[0]
        subject_type_val = getattr(first_triplet, 'subject_type', None)
        return bool(subject_type_val)


def run_all_statistical(data, verbose=False):
    results = {"entities": evaluate_obj_subj(data, verbose=verbose),
            "ontology": profile_ontology(data, include_types=has_types_helper(data), verbose=verbose)}

    if len(data) > 0 and getattr(data[0], 'requested_schema', None):
        results["template_adherence"] = evaluate_template_adherence(data, verbose=verbose)

    return results

def main():
    parser = argparse.ArgumentParser(description="Statistical Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--entities", action="store_true", help="Run only entity distribution")
    parser.add_argument("--ontology", action="store_true", help="Run only ontology profiling")
    parser.add_argument("--template-adherence", action="store_true", help="Run only template adherence")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.entities,
        args.ontology,
        args.template_adherence
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected statistical evaluations...\n")
        if args.entities:
            local_results["entities"] = evaluate_obj_subj(data, verbose=args.verbose)
        if args.ontology:
            local_results["ontology"] = profile_ontology(data, include_types=has_types_helper(data), verbose=args.verbose)
        if args.template_adherence:
            local_results["template_adherence"] = evaluate_template_adherence(data, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all statistical evaluations by default...\n")
        local_results = run_all_statistical(data, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()