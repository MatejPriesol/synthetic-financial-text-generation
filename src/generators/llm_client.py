import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class GroqClient:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.model="llama-3.3-70b-versatile"

    def test_connection(self):
        print("Testing connection...")
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=1
            )
            print("Connection Successful!")
        except Exception as e:
            print(f"Connection Failed: {e}")