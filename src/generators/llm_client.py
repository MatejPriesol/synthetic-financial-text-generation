import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class GroqClient:
    def __init__(self, temperature=1):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        self.model="llama-3.3-70b-versatile"
        self.temperature=temperature

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
                model=self.model,
                temperature=self.temperature
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

# Not usable right now, the models are not good enough
class LocalClient:
    def __init__(self, temperature=1.3):
        self.temperature = temperature
        self.url = "http://localhost:11434/api/chat"

    def generate_message(self, system_prompt, user_prompt):
        payload = {
            "model": "qwen2.5",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_gpu": 20
            },
            "format": "json"
        }

        try:
            response = requests.post(self.url, json=payload)
            response_data = response.json()

            # 1. Check for an error instead of a message
            if "error" in response_data:
                print(f"\n[!] OLLAMA API ERROR: {response_data['error']}")
                return None

            # If no error, return the message
            return response_data["message"]["content"]

        except Exception as e:
            print(f"\n[!] Python Request Error: {e}")
            return None