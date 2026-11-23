from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from src.rag_pipeline import AdvancedRAGPipeline, messages_from_payload

session_histories: Dict[str, List[BaseMessage]] = defaultdict(list)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "frontend" / "index.html"
EXTRACTED_IMAGES_DIR = BASE_DIR / "data" / "extracted_images"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_pipeline = AdvancedRAGPipeline()
    yield


app = FastAPI(
    title="Advanced Multimodal Intent-Routed RAG API",
    version="1.0.0",
    lifespan=lifespan,
)

EXTRACTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/extracted_images", StaticFiles(directory=str(EXTRACTED_IMAGES_DIR)), name="extracted_images")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User query to process.")
    session_id: str = Field(default="default", min_length=1, description="Conversation session ID.")
    history: Optional[List[ChatTurn]] = Field(
        default=None,
        description="Optional client-supplied chat history. If omitted, server session history is used.",
    )


class ChatResponse(BaseModel):
    session_id: str
    intent: Literal["DOCUMENT", "HISTORY", "GENERAL"]
    standalone_query: str
    answer: str
    sources: List[str]
    images: List[str]


@app.get("/", response_class=FileResponse)
def serve_frontend() -> FileResponse:
    if not FRONTEND_FILE.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return FileResponse(path=str(FRONTEND_FILE), media_type="text/html")


@app.get("/health")
def health_check() -> dict:
    pipeline: AdvancedRAGPipeline = app.state.rag_pipeline
    return {
        "status": "ok",
        "llm_model": pipeline.settings.llm_model,
        "embedding_model": pipeline.settings.embedding_model,
        "supabase_table": pipeline.settings.supabase_table,
    }


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    pipeline: AdvancedRAGPipeline = app.state.rag_pipeline

    if payload.history is not None:
        history = messages_from_payload([turn.model_dump() for turn in payload.history])
    else:
        history = session_histories[payload.session_id]

    try:
        result = pipeline.ask(question=payload.question, chat_history=history)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {exc}") from exc

    # Append current turn to server session history if client did not manage history explicitly
    if payload.history is None:
        session_histories[payload.session_id].append(HumanMessage(content=payload.question))
        session_histories[payload.session_id].append(AIMessage(content=result.answer))

    return ChatResponse(
        session_id=payload.session_id,
        intent=result.intent,
        standalone_query=result.standalone_query,
        answer=result.answer,
        sources=result.sources,
        images=result.images,
    )


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    session_histories.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)