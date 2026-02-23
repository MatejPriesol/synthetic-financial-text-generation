import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Triplet:
    subject: str
    relation: str
    object: str

    subject_type: str = None
    object_type: str = None

@dataclass
class FireSample:
    id: int | str
    sentence: str
    triplets: List[Triplet]

def load_fire_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    all_samples = []

    for item in raw_data:
        sentence = " ".join(item['tokens'])

        entities = {i: ent for i, ent in enumerate(item['entities'])}

        extracted_triples = []

        for relation in item.get('relations', []):
            try:
                subj = entities[relation['head']]
                obj = entities[relation['tail']]

                triplet = Triplet(
                    subject=subj['text'],
                    relation=relation['type'],
                    object=obj['text'],

                    subject_type=subj['type'],
                    object_type=obj['type'],
                )

                extracted_triples.append(triplet)
            except KeyError:
                print(f'KeyError: {relation}')
                continue

        sample = FireSample(
            id=item.get('orig_id', -1),
            sentence=sentence,
            triplets=extracted_triples
        )
        all_samples.append(sample)

    return all_samples

def load_fire_relations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    return list(raw_data['relations'].keys())

def load_processed_fire_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Convert to Triplet objects
    samples = []
    for item in raw_data:
        triplets = [Triplet(**t) for t in item['triplets']]
        samples.append(FireSample(id=item['id'], sentence=item['sentence'], triplets=triplets))

    return samples

def load_processed_fire_relations(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        relations = json.load(f)

    return relations

def load_synthetic_fire_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    examples = []

    for i, item in enumerate(raw_data):
        triplets = [Triplet(**t) for t in item.get('triplets', [])]

        example = FireSample(
            id=f"synth_{i}",
            sentence=item.get('text', ''),
            triplets=triplets
        )
        examples.append(example)

    return examples

def main():
    print('Loading data...')
    data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    relations_path = Path("data") / "02_processed" / "fire_types_processed.json"

    data = load_processed_fire_data(data_path)
    relations = load_processed_fire_relations(relations_path)

    print(relations)
    print(f"Sample 2: {data[1].sentence}")
    print(f"Sample 2: {data[1].triplets}")


if __name__ == "__main__":
    main()