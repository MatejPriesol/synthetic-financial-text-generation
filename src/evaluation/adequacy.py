import re
import argparse
import json
import os

from collections import Counter

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
    # Sentence-level trackers (Strict)
    sentence_match_count = 0
    total_valid_sentences = 0

    # Triplet-level trackers (Granular)
    total_requested_triplets = 0
    total_matched_triplets = 0

    if verbose:
        print("\n Template Adherence Mismatches:")

    for sample in data:
        req_schemas = getattr(sample, 'requested_schema', None)

        if not req_schemas:
            continue

        if isinstance(req_schemas, str):
            req_schemas = [req_schemas]

        total_valid_sentences += 1
        total_requested_triplets += len(req_schemas)

        gen_schemas = []
        for t in sample.triplets:
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')
            gen_schemas.append(f"[{s_type}] --({t.relation})--> [{o_type}]")

        req_counter = Counter(req_schemas)
        gen_counter = Counter(gen_schemas)

        # 1. Triplet-Level Math: How many of the specific requested triplets were found?
        # We use min() so if it generates 3 Quantity schemas but we only asked for 2, it only scores 2.
        matched_in_this_sentence = sum(min(gen_counter[schema], req_count) for schema, req_count in req_counter.items())
        total_matched_triplets += matched_in_this_sentence

        # 2. Sentence-Level Math: Did it get ALL of them?
        is_strict_match = all(gen_counter[schema] >= count for schema, count in req_counter.items())

        if is_strict_match:
            sentence_match_count += 1
        else:
            if verbose:
                sentence_text = getattr(sample, 'sentence', getattr(sample, 'text', 'Unknown Text'))
                print(f"FAILED (Got {matched_in_this_sentence}/{len(req_schemas)}):\n  Sentence:  {sentence_text}")
                print(f"  Requested: {req_schemas}")
                print(f"  Generated: {gen_schemas}\n")

    # Calculate final percentages
    strict_score = (sentence_match_count / total_valid_sentences) * 100 if total_valid_sentences > 0 else 0
    yield_score = (total_matched_triplets / total_requested_triplets) * 100 if total_requested_triplets > 0 else 0

    return {
        "strict_sentence_adherence": round(strict_score, 2),
        "triplet_yield_percentage": round(yield_score, 2),
        "total_sentences_checked": total_valid_sentences,
        "total_triplets_requested": total_requested_triplets,
        "total_triplets_matched": total_matched_triplets
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


def evaluate_schema_compliance(data, allowed_type_schemas):
    total_triplets = 0
    compliant_count = 0
    hallucination_counts = {}

    for sample in data:
        for t in sample.triplets:
            total_triplets += 1
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')

            # Reconstruct the type signature
            signature = f"[{s_type}] --({t.relation})--> [{o_type}]"

            if signature in allowed_type_schemas:
                compliant_count += 1
            else:
                hallucination_counts[signature] = hallucination_counts.get(signature, 0) + 1

    # Calculate percentage
    score = round((compliant_count / total_triplets) * 100, 2) if total_triplets > 0 else 0
    sorted_hallucinations = dict(sorted(hallucination_counts.items(), key=lambda item: item[1], reverse=True))

    return {
        "compliance_percentage": score,
        "total_triplets_checked": total_triplets,
        "total_compliant": compliant_count,
        "total_hallucinated": total_triplets - compliant_count,
        "hallucination_frequencies": sorted_hallucinations
    }


def run_all_adequacy(data, allowed_schemas, verbose=False):
    results = {}

    if len(data) > 0 and getattr(data[0], 'requested_schema', None):
        results["template_adherence"] = evaluate_template_adherence(data, verbose=verbose)

    #results["llm_judge"] = evaluate_llm_as_judge(data, verbose=verbose)
    results["surface_form_copying"] = evaluate_surface_form_copying(data)
    results["schema_compliance"] = evaluate_schema_compliance(data, allowed_schemas)

    return results


def main():
    parser = argparse.ArgumentParser(description="Adequacy Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--llm-judge", action="store_true", help="Run only LLM Judge")
    parser.add_argument("--template-adherence", action="store_true", help="Run only template adherence")
    parser.add_argument("--surface-form", action="store_true", help="Run surface form copying eval")
    parser.add_argument("--schema-compliance", action="store_true", help="Run only schema compliance")

    args = parser.parse_args()

    data, sentences = load_data_for_eval(args.data)

    schema_path = "data/03_synthetic/allowed_schemas_cutoff_2.0%.json"
    with open(schema_path, "r", encoding="utf-8") as f:
        allowed_schemas = json.load(f)

    specific_flags = [
        args.llm_judge,
        args.template_adherence,
        args.surface_form,
        args.schema_compliance
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
        if args.schema_compliance:
            local_results["schema_compliance"] = evaluate_schema_compliance(data, allowed_schemas)


    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all adequacy evaluations by default...\n")
        local_results = run_all_adequacy(data, allowed_schemas, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()