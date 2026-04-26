from enum import Enum
import os
import httpx
from pydantic_settings import BaseSettings
from pydantic import Field


class ModelProvider(str, Enum):
    OPENAI_GPT4O = "openai_gpt4o"
    OPENAI_GPT35 = "openai_gpt35"
    OLLAMA_LLAMA3 = "ollama_llama3"
    OLLAMA_MISTRAL = "ollama_mistral"
    OLLAMA_CUSTOM = "ollama_custom"


class EmbeddingProvider(str, Enum):
    OPENAI_SMALL = "openai_small"
    LOCAL_MINILM = "local_minilm"


MODEL_DISPLAY_NAMES: dict[ModelProvider, str] = {
    ModelProvider.OPENAI_GPT4O: "GPT-4o (OpenAI)",
    ModelProvider.OPENAI_GPT35: "GPT-3.5 Turbo (OpenAI)",
    ModelProvider.OLLAMA_LLAMA3: "Llama 3.2 (Ollama)",
    ModelProvider.OLLAMA_MISTRAL: "Mistral (Ollama)",
    ModelProvider.OLLAMA_CUSTOM: "Custom (Ollama)",
}

MODEL_INTERNAL_NAMES: dict[ModelProvider, str] = {
    ModelProvider.OPENAI_GPT4O: "gpt-4o",
    ModelProvider.OPENAI_GPT35: "gpt-3.5-turbo",
    ModelProvider.OLLAMA_LLAMA3: "llama3.2",
    ModelProvider.OLLAMA_MISTRAL: "mistral",
}

# Jailbreak patterns — block these inputs outright
JAILBREAK_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard your instructions",
    "you are now",
    "act as if you have no restrictions",
    "pretend you have no",
    "your new instructions are",
    "system prompt:",
    "forget everything",
]

MEMORY_WINDOW_SIZE = 10


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


settings = Settings()


def check_ollama_health() -> bool:
    try:
        resp = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False
