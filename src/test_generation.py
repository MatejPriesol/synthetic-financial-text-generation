from src.generators.llm_client import GroqClient


def main():
    client = GroqClient()
    client.test_connection()

if __name__ == "__main__":
    main()