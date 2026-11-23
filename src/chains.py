from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from supabase import Client

from src.config import Settings
from src.prompts import (
    answer_prompt,
    contextualize_prompt,
    general_prompt,
    history_prompt,
)


def build_contextualize_chain(llm: BaseChatModel):
    """Rewrite a follow-up question into a standalone question using chat history."""
    chain = contextualize_prompt | llm | StrOutputParser()

    def _rewrite(question: str, chat_history: Sequence[BaseMessage] | None = None) -> str:
        history = list(chat_history or [])
        if not history:
            return question
        rewritten = chain.invoke({"input": question, "chat_history": history[-6:]})
        return rewritten.strip() or question

    return _rewrite


def retrieve_from_supabase(
    query: str,
    supabase: Client,
    embeddings: HuggingFaceEmbeddings,
    settings: Settings,
    top_k: int | None = None,
) -> List[Document]:
    """Query the Supabase match_documents RPC function and return LangChain Documents."""
    k = top_k or settings.top_k
    query_vector = embeddings.embed_query(query)

    response = supabase.rpc(
        settings.supabase_rpc,
        {
            "query_embedding": query_vector,
            "match_count": k,
            "filter": {},
        },
    ).execute()

    rows = response.data or []
    docs: List[Document] = []
    for row in rows:
        docs.append(
            Document(
                page_content=row.get("content", ""),
                metadata=row.get("metadata") or {},
            )
        )
    return docs


def format_retrieved_docs(docs: List[Document]) -> str:
    """Format retrieved chunks (text, tables, image summaries) with source metadata."""
    if not docs:
        return "No relevant context found in the documents."

    formatted_blocks: List[str] = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source", "unknown_source")
        content_type = meta.get("content_type", "text")
        page = meta.get("page")
        page_str = f", Page {page}" if page is not None else ""
        img_info = f", Image: {meta['image_path']}" if meta.get("image_path") else ""

        header = f"[Excerpt {idx} | Source: {source}{page_str} | Type: {content_type}{img_info}]"
        formatted_blocks.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_blocks)


def collect_image_paths(docs: List[Document]) -> List[str]:
    """Collect unique image paths from retrieved image/diagram chunks."""
    paths: List[str] = []
    seen: set[str] = set()
    for doc in docs:
        img_path = (doc.metadata or {}).get("image_path")
        if img_path and img_path not in seen:
            seen.add(img_path)
            paths.append(img_path)
    return paths


def build_answer_chain(llm: BaseChatModel):
    """LCEL chain for answering questions strictly from retrieved document context."""
    return answer_prompt | llm | StrOutputParser()


def build_history_chain(llm: BaseChatModel):
    """LCEL chain for answering questions about conversation history."""
    return history_prompt | llm | StrOutputParser()


def build_general_chain(llm: BaseChatModel):
    """LCEL chain for greetings, small talk, and scope/role queries."""
    return general_prompt | llm | StrOutputParser()