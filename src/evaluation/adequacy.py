"""
Adequacy Evaluation Module.

This module evaluates how accurately generated text satisfies prompt-level constraints,
including template set matching, entity pool adherence, blueprint triplet count compliance,
and ontological schema validity.
"""

import argparse
import json

from collections import Counter
from pathlib import Path

from src.data_loader import load_data_for_eval

def evaluate_template_adherence(data, verbose=False):
    """
    Evaluates per-sentence and per-triplet precision/recall against requested template sets.
    """
    # Sentence-level trackers (Strict)
    sentence_match_count = 0
    total_valid_sentences = 0

    # Triplet-level trackers (Granular)
    total_requested_triplets = 0
    total_matched_triplets = 0
    total_generated_triplets = 0

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

        total_generated_triplets += len(gen_schemas)

        req_counter = Counter(req_schemas)
        gen_counter = Counter(gen_schemas)

        # Triplet-Level: How many of the specific requested triplets were found?
        matched_in_this_sentence = sum(min(gen_counter[schema], req_count) for schema, req_count in req_counter.items())
        total_matched_triplets += matched_in_this_sentence

        # Sentence-Level: Perfect match required
        is_strict_match = (gen_counter == req_counter)

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

    # Precision (What % of the model's output was actually requested?)
    precision_score = (total_matched_triplets / total_generated_triplets) * 100 if total_generated_triplets > 0 else 0

    # Over-generation count
    hallucinated_schema_count = total_generated_triplets - total_matched_triplets

    return {
        "per_sentence_percentage": round(strict_score, 2),
        "per_triplet_recall_percentage": round(yield_score, 2),
        "per_triplet_precision_percentage": round(precision_score, 2),
        "total_sentences_checked": total_valid_sentences,
        "total_triplets_requested": total_requested_triplets,
        "total_triplets_matched": total_matched_triplets,
        "total_unrequested_triplets": hallucinated_schema_count
    }

def evaluate_entity_adherence(data, verbose=False):
    """
    Verifies whether generated entities exist within the requested literal entity pool.
    """
    sentence_match_count = 0
    total_valid_sentences = 0
    total_generated_entities = 0
    total_adherent_entities = 0

    for sample in data:
        allowed_entities = getattr(sample, 'requested_schema', None)
        if not allowed_entities or not isinstance(allowed_entities, list):
            continue

        total_valid_sentences += 1
        allowed_set = set(allowed_entities)
        normalized_allowed_set = {"".join(e.split()).lower() for e in allowed_entities}

        all_sentence_entities_valid = True

        for t in sample.triplets:
            subj = getattr(t, 'subject', '')
            obj = getattr(t, 'object', '')
            total_generated_entities += 2

            norm_subj = "".join(subj.split()).lower()
            norm_obj = "".join(obj.split()).lower()

            s_match = norm_subj in normalized_allowed_set
            o_match = norm_obj in normalized_allowed_set

            if s_match: total_adherent_entities += 1
            if o_match: total_adherent_entities += 1

            if not (s_match and o_match):
                all_sentence_entities_valid = False

        if all_sentence_entities_valid and len(sample.triplets) > 0:
            sentence_match_count += 1
        elif verbose and len(sample.triplets) > 0:
            sentence_text = getattr(sample, 'sentence', getattr(sample, 'text', 'Unknown Text'))
            print(f"FAILED ENTITY CONSTRAINT:\n  Sentence: {sentence_text}")
            print(f"  Allowed Pool: {allowed_entities}")
            print(f"  Generated Entities: {[(getattr(t,'subject',''), getattr(t,'object','')) for t in sample.triplets]}\n")

    strict_score = (sentence_match_count / total_valid_sentences) * 100 if total_valid_sentences > 0 else 0
    token_score = (total_adherent_entities / total_generated_entities) * 100 if total_generated_entities > 0 else 0

    return {
        "per_sentence_percentage": round(strict_score, 2),
        "per_entity_precision_percentage": round(token_score, 2),
        "total_sentences_checked": total_valid_sentences,
        "total_entities_checked": total_generated_entities,
        "total_entities_adherent": total_adherent_entities
    }

def evaluate_blueprint_adherence(data):
    """
    Measures adherence to requested per-sentence triplet count blueprints.
    """
    total = 0
    passed = 0
    for sample in data:
        target = getattr(sample, 'requested_triplet_count', None)
        if target is not None:
            total += 1
            if len(sample.triplets) == target:
                passed += 1

    score = (passed / total) * 100 if total > 0 else 0

    return {"triplet_blueprint_adherence": score}

def evaluate_ontology_compliance(data, allowed_ontology, extended_ontology_template, template=False):
    """
    Evaluates compliance with allowed ontological facts and identifies hallucinated relations.
    """
    total_triplets = 0
    compliant_count = 0
    hallucination_counts = {}

    ontology = extended_ontology_template if template else allowed_ontology

    for sample in data:
        for t in sample.triplets:
            total_triplets += 1
            s_type = getattr(t, 'subject_type', 'UNKNOWN')
            o_type = getattr(t, 'object_type', 'UNKNOWN')

            # Reconstruct the type signature
            signature = f"[{s_type}] --({t.relation})--> [{o_type}]"

            if signature in ontology:
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

def get_type_of_data(data):
    """
    Identifies whether prompt constraints represent template sets or literal entity pools.
    """
    is_template = False
    is_entity = False

    if getattr(data[0], 'requested_schema', None) is not None:
        sample_req = data[0].requested_schema

        if isinstance(sample_req, list) and len(sample_req) > 0:
            if isinstance(sample_req[0], str) and '-->' in sample_req[0]:
                is_template = True
            else:
                is_entity = True

    return is_template, is_entity


def run_all_adequacy(data, allowed_ontology, extended_ontology_template, verbose=False):
    results = {}

    is_template, is_entity = get_type_of_data(data)

    if is_template:
        results["template_adherence"] = evaluate_template_adherence(data, verbose=verbose)

    if is_entity:
        results["entity_adherence"] = evaluate_entity_adherence(data, verbose=verbose)

    results["blueprint_adherence"] = evaluate_blueprint_adherence(data)
    results["ontology_compliance"] = evaluate_ontology_compliance(data, allowed_ontology, extended_ontology_template, is_template)

    return results


def main():
    parser = argparse.ArgumentParser(description="Adequacy Evaluation Local Runner")
    parser.add_argument("--data", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed results")

    # Specific eval flags
    parser.add_argument("--template-adherence", action="store_true", help="Run only template adherence")
    parser.add_argument("--entity-adherence", action="store_true", help="Run only entity adherence")
    parser.add_argument("--ontology-compliance", action="store_true", help="Run only ontology compliance")
    parser.add_argument("--blueprint-adherence", action="store_true", help="Run only triplet blueprint adherence")

    args = parser.parse_args()

    path_parts = Path(args.data).parts
    is_synthetic = "03_synthetic" in path_parts or ("05_archive" in path_parts and "final_fire" not in path_parts)
    if not is_synthetic:
        print(
            "\n[INFO] Non-synthetic dataset detected. Skipping Adequacy evaluation as it requires prompt constraints.")
        return

    data, sentences = load_data_for_eval(args.data)

    allowed_ontology_path = "data/03_synthetic/allowed_ontology.json"
    with open(allowed_ontology_path, "r", encoding="utf-8") as f:
        allowed_ontology = json.load(f)

    template_ontology_path = "data/03_synthetic/extended_ontology_template.json"
    with open(template_ontology_path, "r", encoding="utf-8") as f:
        extended_ontology_template = json.load(f)

    specific_flags = [
        args.template_adherence,
        args.entity_adherence,
        args.ontology_compliance,
        args.blueprint_adherence
    ]

    local_results = {}

    is_template, is_entity = get_type_of_data(data)

    # If any of the specific flags are True, only run those
    if any(specific_flags):
        print("Running selected adequacy evaluations...\n")
        if args.template_adherence and is_template:
            local_results["template_adherence"] = evaluate_template_adherence(data, verbose=args.verbose)
        if args.entity_adherence and is_entity:
            local_results["entity_adherence"] = evaluate_entity_adherence(data, verbose=args.verbose)
        if args.ontology_compliance:
            local_results["ontology_compliance"] = evaluate_ontology_compliance(data, allowed_ontology, extended_ontology_template, is_template)
        if args.blueprint_adherence:
            local_results["blueprint_adherence"] = evaluate_blueprint_adherence(data)

    # If all specific flags are False, default to running everything
    else:
        print("No specific flags detected. Running all adequacy evaluations by default...\n")
        local_results = run_all_adequacy(data, allowed_ontology, extended_ontology_template, verbose=args.verbose)

    print("\n=== LOCAL RUN RESULTS ===")
    print(json.dumps(local_results, indent=2))

if __name__ == "__main__":
    main()
