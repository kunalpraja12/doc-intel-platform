"""Single-turn RAG chat over uploaded document page embeddings."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI

from app.api.schemas.chat import ChatQueryResponse, ChatSource
from app.dependencies import get_db
from db.models.document import Document
from db.repositories.document_repo import DocumentRepository
from pipeline.extract.embeddings import embed_text

load_dotenv()

router = APIRouter(prefix="/chat", tags=["chat"])
repo = DocumentRepository()


def _answer_with_context(question: str, contexts: list[str]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for chat")
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0,
    )
    prompt = """You are a helpful, conversational assistant answering a user's
question about their documents. Answer directly in short, clear, natural
sentences. Do not begin with phrases like "Based on the provided context" or
"According to the documents". If the answer is a single amount or fact, state
it plainly. Avoid bullet points or a formal report unless the question genuinely
calls for a list.

Use only the document context below. If it does not contain enough information,
say so naturally and do not invent details. Sources are returned separately by
the API; do not recite source metadata unless it helps answer the question.

Document context:
---
""" + "\n\n".join(contexts) + f"""
---

Question: {question}
"""
    response = llm.invoke(prompt)
    return _plain_text_content(response.content)


def _plain_text_content(content: Any) -> str:
    """Normalize LangChain text content blocks to the answer string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block
            if isinstance(block, str)
            else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        ]
        answer = "".join(text_parts).strip()
        if answer:
            return answer
    raise RuntimeError("Gemini returned no plain text answer")


@router.post("/query", response_model=ChatQueryResponse)
def query_chat(
    question: str = Body(..., media_type="text/plain"),
    db: Session = Depends(get_db),
) -> ChatQueryResponse:
    try:
        query_embedding = embed_text(question)
        matches = repo.search_embeddings(db, query_embedding, limit=5)
        if not matches:
            return ChatQueryResponse(
                answer="No document embeddings are available for retrieval yet.",
                sources=[],
            )

        contexts: list[str] = []
        sources: list[ChatSource] = []
        for embedding, distance in matches:
            document = db.get(Document, embedding.document_id)
            if not document:
                continue
            page_label = (
                f"page {embedding.page_number}"
                if embedding.page_number is not None
                else "document"
            )
            contexts.append(
                f"Document {document.id} ({document.file_name}), {page_label}, "
                f"similarity distance {distance:.4f}:\n{embedding.chunk_text}"
            )
            sources.append(
                ChatSource(
                    document_id=str(document.id),
                    file_name=document.file_name,
                    page_number=embedding.page_number,
                )
            )
        answer = _answer_with_context(question, contexts)
        return ChatQueryResponse(answer=answer, sources=sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chat query failed: {exc}") from exc
