"""
Statistical Evaluation Module.

This module extracts empirical frequency distributions across generated entities,
subject/object types, relation schemas, and per-sentence triplet counts.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from src.data_loader import load_data_for_eval

ALIAS_MAP_BASIC = {
    # Corporate Titles
    "chief executive officer": "ceo",
    "chief financial officer": "cfo",
    "chief operating officer": "coo",
    "chief technology officer": "cto",

    # Companies
    "apple inc.": "apple",
    "apple inc": "apple",
    "microsoft corporation": "microsoft",
    "alphabet inc.": "alphabet",
    "alphabet inc": "alphabet",
    "johnson & johnson": "johnson and johnson",

    # Financial Terms
    "s&p 500 index": "s&p 500",
    "s&p 500": "s&p 500",
}

ALIAS_MAP_EXTENDED = {
    **ALIAS_MAP_BASIC,  # Inherit basic mappings

    # Generic Placeholders
    "john doe": "[Generic Person Placeholder]",
    "jane doe": "[Generic Person Placeholder]",
    "john smith": "[Generic Person Placeholder]",
    "jane smith": "[Generic Person Placeholder]",

    # Generic Companies
    "techcorp": "[Generic Company Placeholder]",
    "megacorp": "[Generic Company Placeholder]",
    "financialcorp": "[Generic Company Placeholder]",
    "xyz corp": "[Generic Company Placeholder]",
    "xyz corporation": "[Generic Company Placeholder]",
    "abc corp": "[Generic Company Placeholder]",
    "def corp": "[Generic Company Placeholder]",
    "jkl corp": "[Generic Company Placeholder]",
    "mno corp": "[Generic Company Placeholder]",
}

def evaluate_entities(synthetic_data, top_n=10, verbose=False, alias_mode=None):
    """
    Evaluates entity frequency distributions and top entity types across generated samples.
    """

    def normalize_entity(entity):
        cleaned = entity.strip().lower()
        if alias_mode == "extended":
            return ALIAS_MAP_EXTENDED.get(cleaned, cleaned)
        elif alias_mode == "basic":
            return ALIAS_MAP_BASIC.get(cleaned, cleaned)

        return cleaned

    subjects = [normalize_entity(getattr(t, 'subject', '')) for sample in synthetic_data for t in sample.triplets]
    objects = [normalize_entity(getattr(t, 'object', '')) for sample in synthetic_data for t in sample.triplets]

    subject_types = [getattr(t, 'subject_type', 'UNKNOWN') for sample in synthetic_data for t in sample.triplets]
    object_types = [getattr(t, 'object_type', 'UNKNOWN') for sample in synthetic_data for t in sample.triplets]

    # Unified Trackers
    all_entities = subjects + objects
    all_types = subject_types + object_types

    # Totals for percentage math
    total_entities = len(all_entities)
    total_types = len(all_types)
    total_subjects = len(subjects)
    total_objects = len(objects)

    unified_counts = Counter(all_entities)
    type_counts = Counter(all_types)
    subject_counts = Counter(subjects)
    object_counts = Counter(objects)

    # Helper function to format the dictionary output
    def format_with_pct(counter_obj, total):
        return {
            k: {"count": v, "percentage": round((v / total) * 100, 2) if total > 0 else 0}
            for k, v in counter_obj.most_common(top_n)
        }

    if verbose:
        print(f"\n{'Top Unified Entities':<35} | {'Count (Percentage)'}")
        print("-" * 45)
        for ent, count in unified_counts.most_common(top_n):
            pct = (count / total_entities) * 100 if total_entities > 0 else 0
            print(f"{ent:<35} | {count} ({pct:.2f}%)")

    return {
        "top_unified_entities": format_with_pct(unified_counts, total_entities),
        "top_entity_types": format_with_pct(type_counts, total_types),
        "top_subjects_breakdown": format_with_pct(subject_counts, total_subjects),
        "top_objects_breakdown": format_with_pct(object_counts, total_objects)
    }

def profile_ontology(data, include_types=False, top_n=10, verbose=False):
    """
    Profiles the usage frequencies of relation types and full ontological schemas.
    """
    schema_elements = []
    relation_elements = []

    for sample in data:
        for t in sample.triplets:
            relation = getattr(t, 'relation', 'UNKNOWN_RELATION')
            relation_elements.append(relation)

            if include_types:
                s_type = getattr(t, 'subject_type', 'UNKNOWN_TYPE')
                o_type = getattr(t, 'object_type', 'UNKNOWN_TYPE')
                signature = f"[{s_type}] --({relation})--> [{o_type}]"
                schema_elements.append(signature)
            else:
                schema_elements.append(relation)

    schema_counts = Counter(schema_elements)
    relation_counts = Counter(relation_elements)

    total_schemas = len(schema_elements)
    total_relations = len(relation_elements)

    # Helper function for percentages
    def format_with_pct(counter_obj, total):
        return {
            k: {"count": v, "percentage": round((v / total) * 100, 2) if total > 0 else 0}
            for k, v in counter_obj.most_common(top_n)
        }

    if verbose:
        print(f"\n{'Top Schemas':<55} | {'Count (Percentage)'}")
        print("-" * 65)
        for schema, count in schema_counts.most_common(top_n):
            pct = (count / total_schemas) * 100 if total_schemas > 0 else 0
            print(f"{schema:<55} | {count} ({pct:.2f}%)")

    return {
        "total_triplets_extracted": total_schemas,
        "unique_schemas_used": len(schema_counts),
        "unique_relations_used": len(relation_counts),
        "top_schemas": format_with_pct(schema_counts, total_schemas),
        "top_relations": format_with_pct(relation_counts, total_relations)
    }

def evaluate_triplet_density(data, save_plot_path=None, verbose=False):
    """
    Calculates summary statistics and empirical distribution of triplet counts per sentence.
    """
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

    # Generate and save the plot
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

def has_types_helper(data):
    """
    Checks whether extracted sample triplets contain subject and object entity types.
    """
    if len(data) > 0 and len(data[0].triplets) > 0:
        first_triplet = data[0].triplets[0]
        subject_type_val = getattr(first_triplet, 'subject_type', None)
        return bool(subject_type_val)


def run_all_statistical(data, save_name=None, verbose=False, alias_mode=None):
    results = {"entities": evaluate_entities(synthetic_data=data, top_n=10, verbose=verbose, alias_mode=alias_mode),
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
    parser.add_argument("--alias-mode", type=str, choices=["none", "basic", "extended"], default="none",
                        help="Apply alias mapping for entities (default: none)")

    # Specific eval flags
    parser.add_argument("--entities", action="store_true", help="Run only entity distribution")
    parser.add_argument("--ontology", action="store_true", help="Run only ontology profiling")
    parser.add_argument("--triplet-density", action="store_true", help="Run triplet density eval")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    alias_mode = None if args.alias_mode == "none" else args.alias_mode

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
            local_results["entities"] = evaluate_entities(data, top_n=10, verbose=args.verbose, alias_mode=alias_mode)
        if args.ontology:
            local_results["ontology"] = profile_ontology(data, include_types=has_types_helper(data), verbose=args.verbose)
        if args.triplet_density:
            local_results["triplet_density"] = evaluate_triplet_density(data, verbose=args.verbose)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all statistical evaluations by default...\n")
        local_results = run_all_statistical(data, verbose=args.verbose, alias_mode=alias_mode)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()
