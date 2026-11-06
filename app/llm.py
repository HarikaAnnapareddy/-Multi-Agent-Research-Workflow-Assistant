from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from app import config


@lru_cache
def get_llm() -> ChatOllama:
    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)


@lru_cache
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
