import random
import json
import time

from tqdm import tqdm
from pathlib import Path

from src.generators.llm_client import GroqClient
from src.generators.prompt_templates import PromptTemplates
from src.data_loader import load_fire_data, load_fire_types, load_processed_fire_data, load_processed_fire_types


def test_connection():
    client = GroqClient()
    client.test_connection()

def main():
    client = GroqClient()

    data_path = Path("data") / "02_processed" / "fire_train_processed.json"
    relations_path = Path("data") / "02_processed" / "fire_relation_types_processed.json"

    output_dir = Path("data") / "03_synthetic"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "few_shot_650_samples_4_examples.json"


    print("Loading data..")
    data = load_processed_fire_data(data_path)
    relations = load_processed_fire_types(relations_path)
    print(f"Loaded {len(data)} data samples and {len(relations)} relations")

    print("\n--------------------")
    print("Starting batch generation")

    num_of_examples = 4
    num_of_batches = 100
    sentence_per_batch = 10

    system_prompt = PromptTemplates.get_system_prompt()
    task = PromptTemplates.generate_n_sentences(sentence_per_batch)

    all_generated_sentences = []

    for i in tqdm(range(num_of_batches)):
        examples = random.sample(data, num_of_examples)
        rules = PromptTemplates.few_shot(relations, examples)

        response = client.generate_message(system_prompt, rules, task)

        try:
            batch_data = json.loads(response)
            all_generated_sentences.extend(batch_data)
        except json.JSONDecodeError as e:
            print(f"  -> Failed to parse JSON for this batch. Raw output:\n{response}")

        time.sleep(2)

    print(f"\nGeneration complete! Saving {len(all_generated_sentences)} sentences")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_generated_sentences, f, indent=4)

    print(f"Success! Data saved to {output_file}")

if __name__ == "__main__":
    main()