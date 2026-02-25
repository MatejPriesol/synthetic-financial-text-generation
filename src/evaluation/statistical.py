import json
from collections import Counter
from pathlib import Path
from src.data_loader import load_processed_fire_data, load_synthetic_fire_data


def evaluate_relation_distribution(original_data, synthetic_data):
    print("Evaluating relation distribution..\n")

    # Extract all relations
    orig_rels = [t.relation for sample in original_data for t in sample.triplets]
    synth_rels = [t.relation for sample in synthetic_data for t in sample.triplets]

    orig_counts = Counter(orig_rels)
    synth_counts = Counter(synth_rels)

    # Calculate total triplets to compute percentages
    total_orig = sum(orig_counts.values())
    total_gen = sum(synth_counts.values())

    # Get a unique list of all relations found in either dataset
    all_unique_relations = sorted(list(set(orig_rels + synth_rels)))

    # Print a formatted table
    print(f"{'Relation':<25} | {'Original %':<12} | {'Generated %'}")
    print("-" * 55)

    for rel in all_unique_relations:
        orig_pct = (orig_counts.get(rel, 0) / total_orig) * 100 if total_orig > 0 else 0
        gen_pct = (synth_counts.get(rel, 0) / total_gen) * 100 if total_gen > 0 else 0

        print(
            f"{rel:<25} | {orig_pct:>5.1f}% ({orig_counts.get(rel, 0):>3}) | {gen_pct:>5.1f}% ({synth_counts.get(rel, 0):>3})")


def evaluate_obj_subj(synthetic_data, top_n=10):
    print("Evaluating object and subject distributions..\n")

    subjects = [triplet.subject for sample in synthetic_data for triplet in sample.triplets]
    objects = [triplet.object for sample in synthetic_data for triplet in sample.triplets]

    subject_counts = Counter(subjects)
    object_counts = Counter(objects)

    top_subjects = subject_counts.most_common(top_n)
    top_objects = object_counts.most_common(top_n)

    print(f"{'Top Subjects (Heads)':<25} | {'Top Objects (Tails)'}")
    print("-" * 50)

    for i in range(top_n):
        s_str = f"{top_subjects[i][0]} ({top_subjects[i][1]})" if i < len(top_subjects) else ""
        o_str = f"{top_objects[i][0]} ({top_objects[i][1]})" if i < len(top_objects) else ""

        print(f"{s_str:<25} | {o_str}")
    print("\n")



def main():
    original_data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    synthetic_data_path = Path("data") / "03_synthetic" / "few_shot_650_samples_4_examples.json"

    original_data = load_processed_fire_data(original_data_path)
    synthetic_data = load_synthetic_fire_data(synthetic_data_path)

    evaluate_relation_distribution(original_data, synthetic_data)
    evaluate_obj_subj(synthetic_data)

if __name__ == "__main__":
    main()