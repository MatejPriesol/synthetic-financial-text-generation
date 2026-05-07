import re
import argparse
import json
import os

from src.data_loader import load_data_for_eval
from src.generators.llm_client import LocalClientEval
from src.generators.prompt_templates import PromptTemplates


llm_judge = LocalClientEval(
    model="qwen2.5:72b",
    temperature=0.0
)


def get_g_eval_score(triplet, sentence):

    system_prompt = PromptTemplates.adequacy_eval_system_prompt()
    user_prompt = PromptTemplates.adequacy_eval_user_prompt(triplet, sentence)

    # Use your custom client!
    result_text = llm_judge.generate_message(system_prompt, user_prompt)

    if not result_text:
        return 0

    # Extract the integer from the response
    match = re.search(r'\d', result_text)
    if match:
        return int(match.group())
    else:
        return 0


def evaluate_llm_as_judge(data, verbose=False):
    total_sentence_score = 0
    valid_sentences = 0
    triplet_score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    if verbose:
        print(f"\nRunning LLM-as-a-Judge Evaluation using {llm_judge.model}...")

    for sample in data:
        sentence = getattr(sample, 'sentence', None)
        triplets = getattr(sample, 'triplets', [])

        if not sentence or not triplets:
            continue

        sentence_triplet_scores = []

        for triplet in triplets:
            triplet_dict = {
                "subject": getattr(triplet, 'subject', ''),
                "relation": getattr(triplet, 'relation', ''),
                "object": getattr(triplet, 'object', ''),
                "subject_type": getattr(triplet, 'subject_type', 'UNKNOWN'),
                "object_type": getattr(triplet, 'object_type', 'UNKNOWN')
            }

            score = get_g_eval_score(triplet_dict, sentence)

            if score > 0:
                sentence_triplet_scores.append(score)
                triplet_score_distribution[score] = triplet_score_distribution.get(score, 0) + 1

                if verbose and score <= 3:
                    print(f"Low Triplet Score ({score}/5):\n  Sentence: {sentence}\n  Triplet: {triplet_dict}\n")

        if sentence_triplet_scores:
            avg_sentence_score = sum(sentence_triplet_scores) / len(sentence_triplet_scores)
            total_sentence_score += avg_sentence_score
            valid_sentences += 1

    overall_avg = (total_sentence_score / valid_sentences) if valid_sentences > 0 else 0

    return {
        "average_sentence_adequacy": round(overall_avg, 3),
        "triplet_score_distribution": triplet_score_distribution,
        "total_sentences_evaluated": valid_sentences,
        "total_triplets_evaluated": sum(triplet_score_distribution.values())
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


def evaluate_surface_form_copying(data):
    copied_count = 0
    total = 0

    for sample in data:
        for triplet in sample.triplets:
            total += 1
            # Check if the exact relation string is in the sentence
            if getattr(triplet, 'relation', '').lower() in getattr(sample, 'sentence', '').lower():
                copied_count += 1

    # Calculate the float score
    score = round((copied_count / total) * 100, 2) if total > 0 else 0

    return {
        "copying_percentage": score,
        "total_relations_checked": total,
        "total_copied_literally": copied_count
    }


def run_all_adequacy(data, verbose=False):
    results = {}

    if len(data) > 0 and getattr(data[0], 'requested_schema', None):
        results["template_adherence"] = evaluate_template_adherence(data, verbose=verbose)

    #results["llm_judge"] = evaluate_llm_as_judge(data, verbose=verbose)
    results["surface_form_copying"] = evaluate_surface_form_copying(data)

    return results


def main():
    parser = argparse.ArgumentParser(description="Adequacy Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--llm-judge", action="store_true", help="Run only LLM Judge")
    parser.add_argument("--template-adherence", action="store_true", help="Run only template adherence")
    parser.add_argument("--surface-form", action="store_true", help="Run surface form copying eval")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    specific_flags = [
        args.llm_judge,
        args.template_adherence,
        args.surface_form
    ]

    local_results = {}

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected adequacy evaluations...\n")
        if args.llm_judge:
            local_results["llm_judge"] = evaluate_llm_as_judge(data, verbose=args.verbose)
        if args.template_adherence:
            local_results["template_adherence"] = evaluate_template_adherence(data, verbose=args.verbose)
        if args.surface_form:
            local_results["surface_form_copying"] = evaluate_surface_form_copying(data)


    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all adequacy evaluations by default...\n")
        local_results = run_all_adequacy(data, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()