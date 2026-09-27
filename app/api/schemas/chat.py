"""Schemas for single-turn document retrieval chat."""

from pydantic import BaseModel, Field


class ChatQueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class ChatSource(BaseModel):
    document_id: str
    file_name: str
    page_number: int | None = None


class ChatQueryResponse(BaseModel):
    answer: str
    sources: list[ChatSource] = Field(default_factory=list)


ChatRequest = ChatQueryRequest
ChatResponse = ChatQueryResponse
