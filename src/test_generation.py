from src.generators.llm_client import GroqClient
from src.generators.prompt_templates import PromptTemplates
from src.data_loader import load_fire_data, load_fire_relations

import random


def test_connection():
    client = GroqClient()
    client.test_connection()

def main():
    client = GroqClient()

    data_path = "data/01_raw/fire_train.json"
    relations_path = "data/01_raw/fire_types.json"

    print("Loading data")
    data = load_fire_data(data_path)
    relations = load_fire_relations(relations_path)
    print(f"Loaded {len(data)} data samples and {len(relations)} relations")

    print("\n--------------------")
    print("Running zero-shot test")

    rules = PromptTemplates.zero_shot(relations)
    task = PromptTemplates.generate_n_sentences(5)

    #response = client.generate_message(rules, task)

    #print("Here's the response:")
    #print(response)

    print("\n--------------------")
    print("Running few-shot test")

    n = 2
    examples = random.sample(data, n)

    rules = PromptTemplates.few_shot(relations, examples)
    task = PromptTemplates.generate_n_sentences(5)

    response = client.generate_message(rules, task)

    print("RESPONSE:")
    print(response)


if __name__ == "__main__":
    main()