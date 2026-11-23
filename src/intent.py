from typing import Literal, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser

from src.prompts import intent_prompt

IntentCategory = Literal["DOCUMENT", "HISTORY", "GENERAL"]


def _parse_intent_label(raw_output: str) -> IntentCategory:
    cleaned = raw_output.strip().upper()
    for label in ("HISTORY", "DOCUMENT", "GENERAL"):
        if cleaned == label or f"\n{label}" in cleaned or cleaned.startswith(label):
            return label  # type: ignore[return-value]
        if label in cleaned:
            return label  # type: ignore[return-value]
    return "DOCUMENT"


def build_intent_chain(llm: BaseChatModel):
    """Create an LCEL intent classification chain returning 'DOCUMENT', 'HISTORY', or 'GENERAL'."""
    chain = intent_prompt | llm | StrOutputParser()

    def _classify(payload: dict) -> IntentCategory:
        question = payload["input"]
        chat_history: Sequence[BaseMessage] = payload.get("chat_history") or []
        recent_history = list(chat_history)[-6:]
        raw = chain.invoke({"input": question, "chat_history": recent_history})
        return _parse_intent_label(raw)

    return _classify