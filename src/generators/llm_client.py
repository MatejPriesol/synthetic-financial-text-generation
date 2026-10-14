import os
import requests
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class GroqClient:
    def __init__(self, temperature=1):
        self.model="llama-3.3-70b-versatile"
        self.temperature=temperature

        self.api_keys = []
        for i in range(1,4):
            key = os.environ.get(f"GROQ_API_KEY_{i}")
            if key:
                self.api_keys.append(key)

        if not self.api_keys:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.current_key_index = 0
        self._init_client()

    def _init_client(self):
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_keys[self.current_key_index]
        )

    def rotate_key(self):
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print("\n  [!] Rate limit reached, rotating key...")
        self._init_client()

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

        max_retries = len(self.api_keys)

        for i in range(max_retries):

            try:
                response = self.client.chat.completions.create(
                    messages=payload,
                    model=self.model,
                    temperature=self.temperature
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    self.rotate_key()
                    time.sleep(1)
                else:
                    print(f"\n  [!] Groq API Error: {e}")
                    return None


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


class LocalClient:
    def __init__(self, temperature=1.3):
        #self.model = "llama3.3:70b"
        self.model = "qwen2.5:72b"
        self.temperature = temperature
        self.url = "http://localhost:11434/api/chat"
        print(f"Set temperature to {self.temperature}")
        print(f"Using model {self.model}")

    def test_connection(self):
        print(f"Testing local Ollama connection for {self.model}...")
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "Hi"}],
                "options": {"num_ctx": 4096, "max_tokens": 1},
                "stream": False
            }
            response = requests.post(self.url, json=payload, timeout=120)
            if response.status_code == 200:
                print("Connection Successful!")
            else:
                print(f"Connection Failed with status code: {response.status_code}")
        except Exception as e:
            print(f"Connection Failed: {e}")

    def generate_message(self, system_prompt, prompt_part1, prompt_part2=None):
        if prompt_part2:
            full_user_msg = f"{prompt_part1}\n\n{prompt_part2}"
        else:
            full_user_msg = f"{prompt_part1}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_user_msg}
            ],
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": 4096
            },
            #"format": "json"
        }

        try:
            response = requests.post(self.url, json=payload)
            response_data = response.json()

            # 1. Check for an error instead of a message
            if "error" in response_data:
                print(f"\n[!] OLLAMA API ERROR: {response_data['error']}")
                return None

            #prompt_tokens = response_data.get("prompt_eval_count", 0)
            #completion_tokens = response_data.get("eval_count", 0)
            #print(f"  -> Local Usage: {prompt_tokens} Input Tokens | {completion_tokens} Output Tokens")

            # If no error, return the message
            return response_data["message"]["content"]

        except Exception as e:
            print(f"\n[!] Python Request Error: {e}")
            return None

    def print_payload(self, system_prompt, prompt_part1, prompt_part2=None):
        if prompt_part2:
            full_user_msg = f"{prompt_part1}\n\n{prompt_part2}"
        else:
            full_user_msg = f"{prompt_part1}"

        payload = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_user_msg}
        ]

        print(f"\n=== DEBUG: EXACT LOCAL PAYLOAD SENT TO {self.model.upper()} ===")
        for message in payload:
            print(f"\n[{message['role'].upper()}]")
            print(message['content'])
        print("========================================\n")