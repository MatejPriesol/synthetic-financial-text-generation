import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

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


def evaluate_triplet_density(data, save_plot_path=None, verbose=False):
    triplet_counts = [len(getattr(sample, 'triplets', [])) for sample in data]

    if not triplet_counts:
        return {"error": "No valid triplets found in dataset"}

    total_samples = len(triplet_counts)

    # Calculate statistics
    avg_triplets = sum(triplet_counts) / total_samples
    max_triplets = max(triplet_counts)
    min_triplets = min(triplet_counts)

    raw_distribution = dict(Counter(triplet_counts))

    # Sort the dictionary by key
    distribution = {k: raw_distribution[k] for k in sorted(raw_distribution)}

    # Calculate percentages for the JSON
    distribution_percentages = {
        f"{k}_triplets": round((v / total_samples) * 100, 2) for k, v in distribution.items()
    }

    if verbose:
        print(f"\n Triplet Density:")
        print(f"  Average: {avg_triplets:.2f} | Max: {max_triplets}")
        for k, v in distribution.items():
            print(f"  {k} triplets: {v} sentences ({distribution_percentages[f'{k}_triplets']}%)")

    # Generate and save the plot silently
    if save_plot_path:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)

        plt.figure(figsize=(10, 5))
        sns.histplot(triplet_counts, discrete=True, shrink=0.8, color="darkorange")

        plt.title("Distribution of Triplets per Sentence", fontsize=14, pad=15)
        plt.xlabel("Number of Triplets", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(range(min_triplets, max_triplets + 1))

        # Save the plot and close it to free up memory
        plt.savefig(save_plot_path, bbox_inches='tight', dpi=300)
        plt.close()

        if verbose:
            print(f"  Plot saved successfully to: {save_plot_path}")

    return {
        "average_triplets": round(avg_triplets, 2),
        "max_triplets": max_triplets,
        "raw_counts": distribution,
        "percentages": distribution_percentages
    }

# Check if it contains the types (under the assumption that if the first one does, they all do) for the ontology
def has_types_helper(data):
    if len(data) > 0 and len(data[0].triplets) > 0:
        first_triplet = data[0].triplets[0]
        subject_type_val = getattr(first_triplet, 'subject_type', None)
        return bool(subject_type_val)


def run_all_statistical(data, save_name=None, verbose=False):
    results = {"entities": evaluate_obj_subj(data, verbose=verbose),
               "ontology": profile_ontology(data, include_types=has_types_helper(data), verbose=verbose)}

    plot_filepath = f"output/diversity_plots/{save_name}_triplet_density.png" if save_name else None

    results["triplet_density"] = evaluate_triplet_density(
        data,
        save_plot_path=plot_filepath,
        verbose=verbose
    )

    return results

def main():
    parser = argparse.ArgumentParser(description="Statistical Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--entities", action="store_true", help="Run only entity distribution")
    parser.add_argument("--ontology", action="store_true", help="Run only ontology profiling")
    parser.add_argument("--triplet-density", action="store_true", help="Run triplet density eval")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.entities,
        args.ontology,
        args.triplet_density
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected statistical evaluations...\n")
        if args.entities:
            local_results["entities"] = evaluate_obj_subj(data, verbose=args.verbose)
        if args.ontology:
            local_results["ontology"] = profile_ontology(data, include_types=has_types_helper(data), verbose=args.verbose)
        if args.triplet_density:
            local_results["triplet_density"] = evaluate_triplet_density(data, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all statistical evaluations by default...\n")
        local_results = run_all_statistical(data, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()