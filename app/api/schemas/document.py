"""Schemas for document upload and response payloads.

TODO: Define DTOs for upload, metadata, status, and extraction results.
"""

from pydantic import BaseModel, Field


class DocumentUploadRequest(BaseModel):
    file_name: str = Field(..., min_length=1)
    document_type: str | None = None


class DocumentResponse(BaseModel):
    id: str
    file_name: str
    status: str = "uploaded"
