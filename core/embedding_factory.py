from langchain_ollama import OllamaEmbeddings
from config.settings import settings

EMBED_MODEL = "nomic-embed-text"


def get_embedding_model() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=settings.ollama_base_url,
    )
