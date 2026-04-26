from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from config.settings import ModelProvider, MODEL_INTERNAL_NAMES, settings


def get_chat_model(provider: ModelProvider, custom_model_name: str = "") -> BaseChatModel:
    if provider in (ModelProvider.OPENAI_GPT4O, ModelProvider.OPENAI_GPT35):
        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )
        return ChatOpenAI(
            model=MODEL_INTERNAL_NAMES[provider],
            api_key=settings.openai_api_key,
            temperature=0.7,
        )

    if provider == ModelProvider.OLLAMA_CUSTOM:
        if not custom_model_name:
            raise ValueError("Custom Ollama provider requires a model name.")
        model_name = custom_model_name
    else:
        model_name = MODEL_INTERNAL_NAMES[provider]

    return ChatOllama(
        model=model_name,
        base_url=settings.ollama_base_url,
        temperature=0.7,
    )
