import json
import re
import nltk
import random

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt

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
class SentenceSample:
    id: int | str
    sentence: str
    triplets: List[Triplet]
    requested_schema: str = None

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

        sample = SentenceSample(
            id=item.get('orig_id', -1),
            sentence=sentence,
            triplets=extracted_triples
        )
        all_samples.append(sample)

    return all_samples

def load_fire_types(filepath, source):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    return list(raw_data[source].keys())

def load_processed_fire_types(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        types = json.load(f)

    return types


def load_processed_fire_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # Convert to Triplet objects
    samples = []
    for item in raw_data:
        triplets = [Triplet(**t) for t in item['triplets']]
        samples.append(SentenceSample(id=item['id'], sentence=item['sentence'], triplets=triplets))

    return samples

def load_synthetic_fire_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    examples = []

    for i, item in enumerate(raw_data):
        triplets = [Triplet(**t) for t in item.get('triplets', [])]

        example = SentenceSample(
            id=f"synth_{i}",
            sentence=item.get('text', ''),
            triplets=triplets,
            requested_schema = item.get('requested_schema')
        )
        examples.append(example)

    return examples

def load_markdown_data(filepath, name):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    # Remove tables (_START_OF_TABLE_ ..... _END_OF_TABLE_)
    raw_text = re.sub(r'_START_OF_TABLE_.*?_END_OF_TABLE_', '', raw_text, flags=re.DOTALL)

    md = MarkdownIt("commonmark")
    html_content = md.render(raw_text)

    # Extract paragraphs
    soup = BeautifulSoup(html_content, 'html.parser')
    paragraph_tags = soup.find_all('p')

    samples = []
    counter = 0

    for p in paragraph_tags:
        text = p.get_text()

        text = re.sub(r'-\s+', '', text)  # Fix word splits (ser- vices -> services)
        text = re.sub(r'([a-zA-Z])\d+\b', r'\1', text)  # Remove inline refs (emissions4 -> emissions)
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

        sentences = nltk.sent_tokenize(text)

        for s in sentences:
            s = s.strip()

            # Skip some weird parsing errors and format issues
            if len(s.split()) < 5:  # Skip short sentences
                continue
            if not (s[0].isupper() or s[0].isdigit()):  # Skip if it starts with weird symbol
                continue
            if s[-1] not in ".!?":
                continue

            # Create the SentenceSample object
            sample = SentenceSample(
                id=f"{name}_{counter}",
                sentence=s,
                triplets=[]  # No triplets here
            )
            samples.append(sample)
            counter += 1

    return samples


def load_processed_markdown_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        triplets = [Triplet(**t) for t in item.get('triplets', [])]

        # Rebuild the SentenceSample
        samples.append(SentenceSample(
            id=item['id'],
            sentence=item['sentence'],
            triplets=triplets
        ))

    return samples


def load_rebel_data(filepath, name="rebel"):
    samples = []
    counter = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse the single line as a JSON object
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract the raw text block
            text = item.get('text', '')
            if not text:
                continue

            # Clean whitespace and split into sentences
            text = " ".join(text.split())
            sentences = nltk.sent_tokenize(text)

            # Store as SentenceSample with empty triplets
            for s in sentences:
                sample = SentenceSample(
                    id=f"{name}_{counter}",
                    sentence=s,
                    triplets=[]
                )
                samples.append(sample)
                counter += 1

    return samples


def load_processed_rebel_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    samples = []
    for item in raw_data:
        triplets = [Triplet(**t) for t in item.get('triplets', [])]

        # Rebuild the SentenceSample object
        samples.append(SentenceSample(
            id=item['id'],
            sentence=item['sentence'],
            triplets=triplets
        ))

    return samples


def load_rebel_sample(filepath, size=10000):
    samples = load_processed_rebel_data(filepath)

    if size < len(samples):
        return random.sample(samples, size)

    return samples


def main():
    print('Loading data...')
    data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    relations_path = Path("data") / "02_processed" / "fire_relation_types_processed.json"

    data = load_processed_fire_data(data_path)
    relations = load_processed_fire_types(relations_path)

    print(relations)
    print(f"Sample 2: {data[1].sentence}")
    print(f"Sample 2: {data[1].triplets}")



# Load the data for the purpose of evaluation (automatically determine which function to use)
def load_data_for_eval(data_path: str):
    path = Path(data_path)

    # TODO: Update for better distinction, probably after the loading od data is refactored
    if "03_synthetic" in path.parts:
        data = load_synthetic_fire_data(path)
    elif "fire" in path.name:
        data = load_processed_fire_data(path)
    elif "rebel" in path.name:
        data = load_rebel_sample(path)
    else:
        data = load_processed_markdown_data(path)

    sentences = [sample.sentence for sample in data]

    return data, sentences

if __name__ == "__main__":
    main()