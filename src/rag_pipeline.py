from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.chains import (
    build_answer_chain,
    build_contextualize_chain,
    build_general_chain,
    build_history_chain,
    collect_image_paths,
    format_retrieved_docs,
    retrieve_from_supabase,
)
from src.config import Settings, get_embeddings, get_llm, get_settings, get_supabase_client
from src.intent import IntentCategory, build_intent_chain


@dataclass
class RAGResponse:
    answer: str
    intent: IntentCategory
    standalone_query: str
    sources: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


class AdvancedRAGPipeline:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.supabase = get_supabase_client(self.settings)
        self.embeddings = get_embeddings(self.settings)
        self.llm = get_llm(self.settings, temperature=0.0)

        self.intent_classifier = build_intent_chain(self.llm)
        self.contextualize_query = build_contextualize_chain(self.llm)
        self.answer_chain = build_answer_chain(self.llm)
        self.history_chain = build_history_chain(self.llm)
        self.general_chain = build_general_chain(self.llm)

    def ask(
        self,
        question: str,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> RAGResponse:
        history = list(chat_history or [])
        intent: IntentCategory = self.intent_classifier(
            {"input": question, "chat_history": history}
        )

        if intent == "HISTORY":
            answer = self.history_chain.invoke(
                {"input": question, "chat_history": history}
            )
            return RAGResponse(
                answer=answer.strip(),
                intent=intent,
                standalone_query=question,
            )

        if intent == "GENERAL":
            answer = self.general_chain.invoke(
                {"input": question, "chat_history": history}
            )
            return RAGResponse(
                answer=answer.strip(),
                intent=intent,
                standalone_query=question,
            )

        # intent == "DOCUMENT"
        standalone_query = self.contextualize_query(question, history)
        retrieved_docs = retrieve_from_supabase(
            query=standalone_query,
            supabase=self.supabase,
            embeddings=self.embeddings,
            settings=self.settings,
        )
        context_text = format_retrieved_docs(retrieved_docs)
        answer = self.answer_chain.invoke(
            {
                "input": standalone_query,
                "context": context_text,
                "chat_history": history,
            }
        )

        sources: List[str] = []
        seen_sources: set[str] = set()
        for doc in retrieved_docs:
            src = (doc.metadata or {}).get("source")
            if src and src not in seen_sources:
                seen_sources.add(src)
                sources.append(src)

        images = collect_image_paths(retrieved_docs)

        return RAGResponse(
            answer=answer.strip(),
            intent=intent,
            standalone_query=standalone_query,
            sources=sources,
            images=images,
        )


def messages_from_payload(raw_history: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert role/content dicts into LangChain HumanMessage/AIMessage objects."""
    messages: List[BaseMessage] = []
    for item in raw_history:
        role = (item.get("role") or "").strip().lower()
        content = item.get("content") or ""
        if not content:
            continue
        if role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
    return messages