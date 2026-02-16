import json
from dataclasses import dataclass
from typing import List


@dataclass
class Triplet:
    object: str
    relation: str
    subject: str

    object_type: str
    subject_type: str

@dataclass
class FireSample:
    id: int
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
                obj = entities[relation['head']]
                subj = entities[relation['tail']]

                triplet = Triplet(
                    object=obj['text'],
                    relation=relation['type'],
                    subject=subj['text'],

                    object_type=obj['type'],
                    subject_type=subj['type'],
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

if __name__ == "__main__":
    path = "../data/01_raw/fire_train.json"
    try:
        data = load_fire_data(path)
        print(f"Successfully loaded {len(data)} samples.")
        print(f"Sample 2: {data[1].sentence}")
        print(f"Sample 2: {data[1].triplets}")
    except FileNotFoundError:
        print("File not found")