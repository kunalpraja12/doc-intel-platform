"""Gemini embedding helpers for page-level RAG chunks."""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 768


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for embeddings")
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
        output_dimensionality=EMBEDDING_DIMENSION,
    )


def embed_text(text: str) -> list[float]:
    """Embed one chunk or question using the configured Gemini model."""
    return _get_embeddings().embed_query(
        text,
        output_dimensionality=EMBEDDING_DIMENSION,
    )


def page_chunk_text(page_number: int, text: str, fields: dict[str, Any]) -> str:
    """Format page OCR and structured fields into a retrieval-friendly chunk."""
    return (
        f"Page {page_number}\n"
        f"Raw OCR text:\n{text}\n\n"
        f"Structured extracted fields:\n"
        f"{json.dumps(fields, ensure_ascii=False, indent=2)}"
    )
