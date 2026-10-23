"""
Master Evaluation Pipeline.

This script runs the complete evaluation (Grammar, Statistical, Diversity,
and Adequacy) on a target dataset and aggregates the results into a
final JSON report.
"""

import argparse
import json
import torch
import transformers

from pathlib import Path

from src.data_loader import load_data_for_eval
from src.evaluation.grammatical_correctness import run_all_grammar
from src.evaluation.statistical import run_all_statistical
from src.evaluation.diversity import run_all_diversity
from src.evaluation.adequacy import run_all_adequacy, get_type_of_data

# Silence Hugging Face's warnings
transformers.logging.set_verbosity_error()

def main():
    parser = argparse.ArgumentParser(description="Master Evaluation Runner")

    # Path to the data
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset to evaluate")

    # Which evals to run
    parser.add_argument("--all", action="store_true", help="Run ALL evaluations")
    parser.add_argument("--grammar", action="store_true", help="Run Grammar eval")
    parser.add_argument("--stats", action="store_true", help="Run Statistical evals")
    parser.add_argument("--diversity", action="store_true", help="Run Diversity evals")
    parser.add_argument("--adequacy", action="store_true", help="Run Adequacy evals")

    # Include detailed prints
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed evaluation results")

    # Save results
    parser.add_argument("--save", "-s", type=str, help="Save evaluation results. Provide the name of the file")

    # Name for visualization purposes
    parser.add_argument("--run_name", "-n", type=str, help="A short name for this experiment (Required if saving")

    args = parser.parse_args()

    # Name required if the results are being saved
    if args.save and not args.run_name:
        parser.error("--run_name is required when --save is active!")

    print(f"Loading data from: {args.data}")
    data, sentences = load_data_for_eval(args.data)
    print(f"Loaded {len(sentences)} sentences.\n")

    schema_path = "data/03_synthetic/allowed_schemas_cutoff_2.0%.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        allowed_schemas = json.load(f)

    template_schema_path = "data/03_synthetic/allowed_schemas_template.json"
    with open(template_schema_path, "r", encoding="utf-8") as f:
        allowed_schemas_template = json.load(f)

    is_template, is_entity = get_type_of_data(data)

    schemas_to_save = allowed_schemas_template if is_template else allowed_schemas

    if "02_processed" in Path(args.data).parts:
        schemas_to_save = []

    if args.verbose:
        print(f"Verbose flag enabled, detailed evaluation results will be printed.\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Running evaluations... (Might take a while based on dataset size). Using " + device.upper())

    final_results = {
        "Overview": {
            "general": {
                "run_name": args.run_name,
                "dataset_size": len(sentences),
                "device_used": device.upper(),
                "dataset_path": args.data,
                "allowed_schemas": schemas_to_save
            }
        }
    }

    if args.all or args.grammar:
        if args.verbose:
            print("\n\n=== Running Grammar Evals ===\n")
        else:
            print("Running Grammar Evals ...")

        final_results["Grammar"] = run_all_grammar(sentences, args.verbose)

    if args.all or args.stats:
        if args.verbose:
            print("\n\n=== Running Statistical Evals ===\n")
        else:
            print("Running Statistical Evals...")

        final_results["Statistical"] = run_all_statistical(data, args.save, args.verbose)

    if args.all or args.diversity:
        if args.verbose:
            print("\n\n=== Running Diversity Evals ===\n")
        else:
            print("Running Diversity Evals...")

        final_results["Diversity"] = run_all_diversity(sentences, args.verbose)

    path_parts = Path(args.data).parts
    if (args.all or args.adequacy) and ("03_synthetic" in path_parts or "05_archive" in path_parts):
        if args.verbose:
            print("\n\n=== Running Adequacy Evals ===\n")
        else:
            print("Running Adequacy Evals...")

        final_results["Adequacy"] = run_all_adequacy(data, allowed_schemas, allowed_schemas_template, args.verbose)


    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    for module_name, metrics in final_results.items():
        print(f"\n--- {module_name.upper()} METRICS ---")

        # Iterate through the returned dictionaries and print them
        for sub_module, data in metrics.items():
            print(f"\n  [{sub_module.replace('_', ' ').title()}]")
            for key, value in data.items():
                clean_key = key.replace("_", " ").title()

                # Check if it's a percentage to format it cleanly
                if "ratio" in key.lower() or "percentage" in key.lower():
                    print(f"{clean_key:<35}: {value}%")
                else:
                    print(f"{clean_key:<35}: {value}")

    print("\n" + "=" * 50)

    if args.save:
        save_path = f"data/04_reports/{args.save}.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main()
