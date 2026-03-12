import os
import json
from dotenv import load_dotenv
from openai import OpenAI

from src.generators.prompt_templates import PromptTemplates

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


    def generate_message(self, system_prompt, prompt_part1, prompt_part2=None):
        if prompt_part2:
            full_user_msg = f"{prompt_part1}\n\n{prompt_part2}"
        else:
            full_user_msg = f"{prompt_part1}"

        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_msg}
        ]

        try:
            response = self.client.chat.completions.create(
                messages=payload,
                model=self.model
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating message: {e}"


    def print_payload(self, system_prompt, prompt_part1, prompt_part2=None):
        if prompt_part2:
            full_user_msg = f"{prompt_part1}\n\n{prompt_part2}"
        else:
            full_user_msg = f"{prompt_part1}"

        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_msg}
        ]

        print("\n=== DEBUG: EXACT PAYLOAD SENT TO LLM ===")
        for message in payload:
            print(f"\n[{message['role'].upper()}]")
            print(message['content'])
        print("========================================\n")
