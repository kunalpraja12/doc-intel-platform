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
import logging

from app.api.schemas.document import (
    DocumentResponse,
    DocumentDetailResponse,
    ExtractionResultResponse,
    StructuredFieldsResponse,
    StructuredLineItemResponse,
    PageExtractionResponse,
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
logger = logging.getLogger(__name__)

SUPPORTED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/pdf",
}
SUPPORTED_CONTENT_TYPES_MESSAGE = ", ".join(sorted(SUPPORTED_CONTENT_TYPES))


def _render_pdf_pages(pdf_bytes: bytes, output_dir: Path, file_stem: str) -> list[Path]:
    """Render every PDF page to a PNG in one pass over the opened PDF."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError(
            "PDF uploads require PyMuPDF. Install it with: pip install PyMuPDF"
        ) from exc

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf:
            page_count = pdf.page_count
            if page_count == 0:
                raise ValueError("The uploaded PDF contains no pages")
            page_paths: list[Path] = []
            for page_number in range(page_count):
                page = pdf.load_page(page_number)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                page_path = output_dir / f"{file_stem}_page{page_number + 1}.png"
                pixmap.save(str(page_path))
                page_paths.append(page_path)
            logger.info("Processed %s pages from PDF", page_count)
            return page_paths
    except HTTPException:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to convert PDF pages to images: {exc}") from exc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)) -> list[DocumentResponse]:
    stmt = select(Document)
    results = db.execute(stmt).scalars().all()
    return [DocumentResponse(id=str(d.id), file_name=d.file_name, status=d.status) for d in results]


@router.post("/upload", response_model=DocumentDetailResponse)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> DocumentDetailResponse:
    if file.content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                f"Supported types: {SUPPORTED_CONTENT_TYPES_MESSAGE}"
            ),
        )

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

    # Convert PDF pages to images, then run the same OCR pipeline used for
    # image uploads. Non-PDF uploads retain the existing single-image path.
    try:
        if file.content_type == "application/pdf":
            page_paths = _render_pdf_pages(
                contents,
                UPLOAD_DIR,
                f"{timestamp}_{Path(safe_name).stem}",
            )
            extractor = OCRExtractor()
            page_texts: list[str] = []
            page_words: list[dict] = []
            for page_number, page_path in enumerate(page_paths, start=1):
                page_result = extractor.extract_from_path(
                    str(page_path),
                    use_preprocess=True,
                )
                page_texts.append(page_result.get("text", ""))
                page_words.extend(
                    {**word, "page": page_number}
                    for word in page_result.get("words", [])
                )
            extracted = {
                "text": "\n\n".join(
                    f"--- Page {page_number} ---\n\n{text}"
                    for page_number, text in enumerate(page_texts, start=1)
                ),
                "words": page_words,
            }
            page_texts_for_extraction = page_texts
        else:
            extractor = OCRExtractor()
            # Use preprocessing by default; the extractor will auto-upscale small images
            extracted = extractor.extract_from_path(str(dest_path), use_preprocess=True)
            page_texts_for_extraction = [extracted.get("text", "")]

        page_results = [
                {
                    "page_number": page_number,
                    "text": page_text,
                    "fields": None,
                }
                for page_number, page_text in enumerate(page_texts_for_extraction, start=1)
        ]

        # Persist raw OCR before field extraction updates, so raw text remains
        # available if a per-page Gemini call fails.
        raw_data = {
            "page_count": len(page_results),
            "pages": page_results,
            "text": extracted.get("text", ""),
            "words": extracted.get("words", []),
        }

        extraction_result = repo.add_extraction_result(
            db=db,
            document_id=doc.id,
            extracted_data=raw_data,
            model_name="tesseract",
        )

        # Gemini is intentionally called once per page. This is synchronous and
        # preserves each page's own uncertainty and line-item context.
        for page in page_results:
            page["fields"] = extract_fields(page["text"]).model_dump()

        fields_data = page_results[0]["fields"]
        combined_data = {
            "page_count": len(page_results),
            "pages": page_results,
            "text": extracted.get("text", ""),
            "words": extracted.get("words", []),
            "fields": fields_data,
        }
        repo.update_extraction_data(db=db, extraction_id=extraction_result.id, extracted_data=combined_data)
        repo.add_line_items(
            db=db,
            document_id=doc.id,
            line_items=[
                item
                for page in page_results
                for item in page["fields"].get("line_items", [])
            ],
        )
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
        page_count=combined_data["page_count"],
        pages=_pages_response(combined_data.get("pages", [])),
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
        page_count=latest.extracted_data.get("page_count", 1) if latest and latest.extracted_data else 1,
        pages=_pages_response(latest.extracted_data.get("pages", []) if latest and latest.extracted_data else []),
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


def _pages_response(pages: list[dict]) -> list[PageExtractionResponse]:
    return [
        PageExtractionResponse(
            page_number=page["page_number"],
            text=page.get("text", ""),
            fields=StructuredFieldsResponse(
                **{
                    **page["fields"],
                    "line_items": [
                        StructuredLineItemResponse(**item)
                        for item in page["fields"].get("line_items", [])
                    ],
                }
            ) if page.get("fields") is not None else None,
        )
        for page in pages
    ]
