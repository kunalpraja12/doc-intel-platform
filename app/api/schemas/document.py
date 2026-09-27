"""Schemas for document upload and response payloads.

DTOs used by the documents endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    file_name: str = Field(..., min_length=1)
    document_type: str | None = None


class ExtractionWord(BaseModel):
    text: str
    page: Optional[int] = None
    left: int
    top: int
    width: int
    height: int
    conf: Optional[float]


class ExtractionResultResponse(BaseModel):
    text: str
    words: List[ExtractionWord] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    model_name: Optional[str] = None
    fields: Optional["StructuredFieldsResponse"] = None


class StructuredLineItemResponse(BaseModel):
    description: Optional[str] = None
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    total: Optional[float] = None


class StructuredFieldsResponse(BaseModel):
    vendor_name: Optional[str] = None
    document_type: Optional[str] = None
    total_amount: Optional[float] = None
    date: Optional[str] = None
    invoice_or_receipt_number: Optional[str] = None
    line_items: List[StructuredLineItemResponse] = Field(default_factory=list)


class PageExtractionResponse(BaseModel):
    page_number: int
    text: str
    fields: Optional[StructuredFieldsResponse] = None


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    status: str = "uploaded"


class DocumentDetailResponse(DocumentResponse):
    duplicate: bool = False
    file_path: Optional[str] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    page_count: int = 1
    pages: List[PageExtractionResponse] = Field(default_factory=list)
    extraction: Optional[ExtractionResultResponse] = None


class UploadFileResult(BaseModel):
    file_name: str
    status: str
    duplicate: bool = False
    error: Optional[str] = None
    id: Optional[str] = None
    file_path: Optional[str] = None
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    page_count: int = 1
    pages: List[PageExtractionResponse] = Field(default_factory=list)
    extraction: Optional[ExtractionResultResponse] = None
