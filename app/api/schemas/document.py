"""Schemas for document upload and response payloads.

DTOs used by the documents endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    file_name: str = Field(..., min_length=1)
    document_type: str | None = None


class ExtractionWord(BaseModel):
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: Optional[float]


class ExtractionResultResponse(BaseModel):
    text: str
    words: List[ExtractionWord] = []
    confidence_score: Optional[float] = None
    model_name: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    status: str = "uploaded"


class DocumentDetailResponse(DocumentResponse):
    file_path: Optional[str] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    extraction: Optional[ExtractionResultResponse] = None
