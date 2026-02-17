from src.generators.llm_client import GroqClient
from src.generators.prompt_templates import PromptTemplates
from src.data_loader import load_fire_data, load_relations


def test_connection():
    client = GroqClient()
    client.test_connection()

def main():
    client = GroqClient()

    data_path = "data/01_raw/fire_train.json"
    relations_path = "data/01_raw/fire_types.json"

    print("Loading data")
    data = load_fire_data(data_path)
    relations = load_relations(relations_path)
    print(f"Loaded {len(data)} data samples and {len(relations)} relations")

    print("\n--------------------")
    print("Running zero-shot test")

    rules = PromptTemplates.zero_shot(relations)
    task = PromptTemplates.five_unique_sentences()

    response = client.generate_message(rules, task)

    print("Here's the response:")
    print(response)

if __name__ == "__main__":
    main()