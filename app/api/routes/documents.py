"""Document upload and metadata routes.

Endpoints:
- GET /documents - list documents
- POST /documents/upload - upload a file, save to uploads/, run OCR synchronously
- GET /documents/{document_id} - return document status + latest extraction
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session
from pathlib import Path
from datetime import datetime
import os

from app.api.schemas.document import (
    DocumentResponse,
    DocumentDetailResponse,
    ExtractionResultResponse,
    StructuredFieldsResponse,
    StructuredLineItemResponse,
)
from app.dependencies import get_db
# Ensure all models are imported so relationships resolve
import db.models  # noqa: F401
from db.models.document import Document
from db.repositories.document_repo import DocumentRepository

# Import OCR extractor
from pipeline.extract.ocr import OCRExtractor
from pipeline.extract.field_extraction import extract_fields

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path(os.environ.get("UPLOADS_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

repo = DocumentRepository()


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)) -> list[DocumentResponse]:
    stmt = select(Document)
    results = db.execute(stmt).scalars().all()
    return [DocumentResponse(id=str(d.id), file_name=d.file_name, status=d.status) for d in results]


@router.post("/upload", response_model=DocumentDetailResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> DocumentDetailResponse:
    # Save uploaded file to uploads/ with a timestamped name
    safe_name = Path(file.filename).name
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest_name = f"{timestamp}_{safe_name}"
    dest_path = UPLOAD_DIR / dest_name

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to save uploaded file: {exc}")

    # Create document record with status='pending'
    doc = repo.create_document(
        db=db,
        file_name=safe_name,
        file_path=str(dest_path),
        content_type=file.content_type,
        file_size=len(contents),
        status="pending",
    )

    # Run OCR synchronously (blocking). If OCR fails, mark as failed.
    try:
        extractor = OCRExtractor()
        # Use preprocessing by default; the extractor will auto-upscale small images
        extracted = extractor.extract_from_path(str(dest_path), use_preprocess=True)

        # Persist raw OCR before calling the LLM, so the raw text is retained
        # even when field extraction fails.
        extraction_result = repo.add_extraction_result(
            db=db,
            document_id=doc.id,
            extracted_data={"text": extracted.get("text", ""), "words": extracted.get("words", [])},
            model_name="tesseract",
        )
        fields = extract_fields(extracted.get("text", ""))
        fields_data = fields.model_dump()
        combined_data = {
            "text": extracted.get("text", ""),
            "words": extracted.get("words", []),
            "fields": fields_data,
        }
        repo.update_extraction_data(db=db, extraction_id=extraction_result.id, extracted_data=combined_data)
        repo.add_line_items(db=db, document_id=doc.id, line_items=fields_data["line_items"])
        repo.update_status(db=db, document_id=doc.id, status="completed")

        extraction_response = _extraction_response(combined_data, "tesseract+gemini")
    except Exception as exc:
        # Persist failure state and return
        repo.update_status(db=db, document_id=doc.id, status="failed")
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}")

    return DocumentDetailResponse(
        id=str(doc.id),
        file_name=doc.file_name,
        status=doc.status,
        file_path=doc.file_path,
        content_type=doc.content_type,
        file_size=doc.file_size,
        extraction=extraction_response,
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    doc = repo.get_document_with_latest_extraction(db=db, document_id=document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    latest = repo.get_latest_extraction(db=db, document_id=document_id)
    extraction_response = None
    if latest and latest.extracted_data:
        extraction_response = _extraction_response(
            latest.extracted_data,
            latest.model_name,
            latest.confidence_score,
        )

    return DocumentDetailResponse(
        id=str(doc.id),
        file_name=doc.file_name,
        status=doc.status,
        file_path=doc.file_path,
        content_type=doc.content_type,
        file_size=doc.file_size,
        extraction=extraction_response,
    )


def _extraction_response(
    data: dict,
    model_name: str | None,
    confidence_score: float | None = None,
) -> ExtractionResultResponse:
    fields = data.get("fields")
    structured_fields = None
    if fields:
        structured_fields = StructuredFieldsResponse(
            **{
                **fields,
                "line_items": [
                    StructuredLineItemResponse(**item)
                    for item in fields.get("line_items", [])
                ],
            }
        )
    return ExtractionResultResponse(
        text=data.get("text", ""),
        words=data.get("words", []),
        confidence_score=confidence_score,
        model_name=model_name,
        fields=structured_fields,
    )
