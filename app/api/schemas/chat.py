"""Schemas for chat prompts and responses.

TODO: Define request/response DTOs for retrieval-augmented Q&A.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []
