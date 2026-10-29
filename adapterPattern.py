from abc import ABC, abstractmethod
from google import genai
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

load_dotenv()

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Passa un prompt, restituisce il testo della risposta."""
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4o", generation_config:dict = {}):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.generation_config = generation_config

    def generate(self, prompt: str) -> dict:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.generation_config
        )
        elapsed = time.perf_counter() - start
        return {"output":resp.choices[0].message.content, "latency":elapsed}

class GeminiClient(LLMClient):
    def __init__(self,  model: str = "gemini-flash-lite-latest", generation_config:dict = {}):
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = f"models/{model}"
        self.generation_config = {
            'max_output_tokens': generation_config["max_tokens"],
            "temperature":generation_config["temperature"]
        }

    def generate(self, prompt: str) -> dict:
        start = time.perf_counter()
        resp = self.client.interactions.create(
            model=self.model,
            input=prompt,
            generation_config=self.generation_config
        )
        elapsed = time.perf_counter() - start
        return {"output":resp.output_text, "latency": elapsed}

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llama3.1:8b", generation_config:dict = {}):
        self.client = OpenAI(base_url=base_url, api_key=os.getenv("OLLAMA_API_KEY"))
        self.model = model
        self.generation_config = generation_config

    def generate(self, prompt: str) -> dict:
        start = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **self.generation_config
        )
        elapsed = time.perf_counter() - start
        return {"output":resp.choices[0].message.content, "latency":elapsed}

def get_client(provider: str, **kwargs) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "ollama":
        return OllamaClient(**kwargs)
    else:
        raise ValueError(f"Provider sconosciuto: {provider}")