import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import Client, create_client

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str
    supabase_url: str
    supabase_key: str
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    llm_model: str = "openai/gpt-oss-20b"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    supabase_table: str = "documents"
    supabase_rpc: str = "match_documents"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5


def get_settings() -> Settings:
    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
    supabase_url = os.getenv("SUPABASE_URL", "").strip()
    supabase_key = os.getenv("SUPABASE_KEY", "").strip()

    missing = [
        name
        for name, val in [
            ("GROQ_API_KEY", groq_api_key),
            ("SUPABASE_URL", supabase_url),
            ("SUPABASE_KEY", supabase_key),
        ]
        if not val
    ]
    if missing:
        raise ValueError(f"Missing required environment variables in .env: {', '.join(missing)}")

    return Settings(
        groq_api_key=groq_api_key,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip(),
        llm_model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b").strip(),
        vision_model=os.getenv("VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").strip(),
        supabase_table=os.getenv("SUPABASE_TABLE", "documents").strip(),
        supabase_rpc=os.getenv("SUPABASE_RPC", "match_documents").strip(),
    )


def get_supabase_client(settings: Settings) -> Client:
    return create_client(settings.supabase_url, settings.supabase_key)


def get_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_llm(settings: Settings, temperature: float = 0.0) -> ChatGroq:
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=temperature,
    )