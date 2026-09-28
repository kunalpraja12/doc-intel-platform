"""Document upload and metadata routes.

Endpoints:
- GET /documents - list documents
- POST /documents/upload - upload a file, save to uploads/, run OCR synchronously
- GET /documents/{document_id} - return document status + latest extraction
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from pathlib import Path
import hashlib
import os
import logging
import uuid

from app.api.schemas.document import (
    DocumentResponse,
    DocumentDetailResponse,
    ExtractionResultResponse,
    StructuredFieldsResponse,
    StructuredLineItemResponse,
    PageExtractionResponse,
    UploadFileResult,
)
from app.core.errors import (
    FILE_MESSAGE,
    FILE_TOO_LARGE_MESSAGE,
    user_friendly_message,
)
from app.dependencies import get_db
# Ensure all models are imported so relationships resolve
import db.models  # noqa: F401
from db.models.document import Document
from db.models.extraction import ExtractionResult
from db.repositories.document_repo import DocumentRepository
from db.session import SessionLocal

# Import OCR extractor
from pipeline.extract.ocr import OCRExtractor
from pipeline.extract.field_extraction import extract_fields, validate_line_items
from pipeline.extract.embeddings import embed_text, page_chunk_text

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
MAX_FILE_SIZE = 20 * 1024 * 1024
CONTENT_TYPE_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".pdf": "application/pdf",
}


def _render_pdf_pages(
    pdf_bytes: bytes,
    output_dir: Path,
    file_stem: str,
) -> list[tuple[Path, str]]:
    """Render pages and collect embedded text while the PDF is open."""
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
            page_results: list[tuple[Path, str]] = []
            for page_number in range(page_count):
                page = pdf.load_page(page_number)
                pixmap = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
                page_path = output_dir / f"{file_stem}_page{page_number + 1}.png"
                pixmap.save(str(page_path))
                page_results.append((page_path, page.get_text("text").strip()))
            logger.info("Processed %s pages from PDF", page_count)
            return page_results
    except HTTPException:
        raise
    except Exception as exc:
        raise RuntimeError(f"Failed to convert PDF pages to images: {exc}") from exc


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)) -> list[DocumentResponse]:
    stmt = select(Document)
    results = db.execute(stmt).scalars().all()
    return [DocumentResponse(id=str(d.id), file_name=d.file_name, status=d.status) for d in results]


@router.post("/upload", response_model=list[UploadFileResult])
async def upload_document(
    files: list[UploadFile] = File(...),
) -> list[UploadFileResult]:
    results: list[UploadFileResult] = []
    for uploaded_file in files:
        filename = Path(uploaded_file.filename or "uploaded-file").name
        try:
            if filename.casefold().endswith(".zip") or uploaded_file.content_type == "application/zip":
                results.append(
                    _upload_error(
                        filename,
                        FILE_MESSAGE,
                    )
                )
                continue

            if uploaded_file.size is not None and uploaded_file.size > MAX_FILE_SIZE:
                results.append(_upload_error(filename, FILE_TOO_LARGE_MESSAGE))
                continue

            contents = await uploaded_file.read()
            if len(contents) > MAX_FILE_SIZE:
                results.append(_upload_error(filename, FILE_TOO_LARGE_MESSAGE))
                continue

            content_type = _content_type(filename, uploaded_file.content_type)
            if content_type not in SUPPORTED_CONTENT_TYPES:
                results.append(
                    _upload_error(
                        filename,
                        FILE_MESSAGE,
                    )
                )
                continue
            results.append(_process_document_upload(contents, filename, content_type))
        except Exception as exc:
            logger.exception("Failed to process upload %s", filename)
            results.append(
                _upload_error(
                    filename,
                    user_friendly_message(exc, ai_operation=True, upload=True),
                )
            )
        finally:
            await uploaded_file.close()
    return results


def _upload_error(filename: str, message: str) -> UploadFileResult:
    return UploadFileResult(
        file_name=filename,
        status=(
            "skipped"
            if message == FILE_TOO_LARGE_MESSAGE
            else "failed"
        ),
        message=message,
    )


def _content_type(filename: str, provided_type: str | None) -> str | None:
    return CONTENT_TYPE_BY_SUFFIX.get(Path(filename).suffix.casefold(), provided_type)


def _document_needs_review(page_results: list[dict]) -> bool:
    invoice_total = next(
        (
            page["fields"].get("total_amount")
            for page in page_results
            if page["fields"].get("total_amount") is not None
        ),
        None,
    )
    line_net_amounts = [
        (
            item.get("net_amount")
            if item.get("net_amount") is not None
            else item.get("total")
        )
        for page in page_results
        for item in page["fields"].get("line_items", [])
    ]
    if (
        invoice_total is None
        or not line_net_amounts
        or any(amount is None for amount in line_net_amounts)
    ):
        return False

    net_sum = sum(line_net_amounts)
    return abs(net_sum - invoice_total) > max(abs(invoice_total) * 0.02, 0.01)


def _process_document_upload(
    contents: bytes,
    filename: str,
    content_type: str,
) -> UploadFileResult:
    safe_name = Path(filename).name
    content_hash = hashlib.sha256(contents).hexdigest()
    with SessionLocal() as read_db:
        existing_doc = repo.get_document_by_content_hash(read_db, content_hash)
        if existing_doc:
            latest = repo.get_latest_extraction(read_db, existing_doc.id)
            detail = _document_detail_response(existing_doc, latest, duplicate=True)
            return UploadFileResult(**detail.model_dump())

    unique_prefix = uuid.uuid4().hex
    dest_path = UPLOAD_DIR / f"{unique_prefix}_{safe_name}"
    dest_path.write_bytes(contents)
    page_paths: list[Path] = []

    try:
        if content_type == "application/pdf":
            rendered_pages = _render_pdf_pages(
                contents,
                UPLOAD_DIR,
                f"{unique_prefix}_{Path(safe_name).stem}",
            )
            extractor = OCRExtractor()
            page_texts: list[str] = []
            page_words: list[dict] = []
            page_paths = [page_path for page_path, _ in rendered_pages]
            for page_number, (page_path, embedded_text) in enumerate(
                rendered_pages,
                start=1,
            ):
                if len(embedded_text) > 50:
                    page_texts.append(embedded_text)
                    logger.info(
                        "Using embedded text for PDF page %s (more than 50 characters)",
                        page_number,
                    )
                else:
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
                    f"--- Page {number} ---\n\n{text}"
                    for number, text in enumerate(page_texts, start=1)
                ),
                "words": page_words,
            }
        else:
            image_result = OCRExtractor().extract_from_path(
                str(dest_path),
                use_preprocess=True,
            )
            page_texts = [image_result.get("text", "")]
            extracted = {
                "text": page_texts[0],
                "words": image_result.get("words", []),
            }

        image_paths = page_paths if content_type == "application/pdf" else [dest_path]
        page_results = [
            {"page_number": number, "text": text, "fields": None}
            for number, text in enumerate(page_texts, start=1)
        ]
        for index, page in enumerate(page_results):
            page_fields = extract_fields(
                page["text"],
                image_path=image_paths[index],
            )
            page["fields"] = validate_line_items(
                page_fields,
                page["text"],
                image_paths[index],
            ).model_dump()

        combined_data = {
            "page_count": len(page_results),
            "pages": page_results,
            "text": extracted["text"],
            "words": extracted["words"],
            "fields": page_results[0]["fields"],
        }
        combined_data["needs_review"] = _document_needs_review(page_results)
        embeddings = []
        for page in page_results:
            chunk = page_chunk_text(page["page_number"], page["text"], page["fields"])
            embeddings.append(
                {
                    "page_number": (
                        page["page_number"]
                        if content_type == "application/pdf"
                        else None
                    ),
                    "embedding": embed_text(chunk),
                    "chunk_text": chunk,
                }
            )
        line_items = [
            item
            for page in page_results
            for item in page["fields"].get("line_items", [])
        ]

        # All OCR and external AI work is complete before opening the write session.
        with SessionLocal() as write_db:
            try:
                with write_db.begin():
                    concurrent_doc = repo.get_document_by_content_hash(
                        write_db,
                        content_hash,
                    )
                    if concurrent_doc:
                        latest = repo.get_latest_extraction(write_db, concurrent_doc.id)
                        detail = _document_detail_response(
                            concurrent_doc,
                            latest,
                            duplicate=True,
                        )
                        result = UploadFileResult(**detail.model_dump())
                    else:
                        doc, extraction = repo.persist_processed_upload(
                            write_db,
                            file_name=safe_name,
                            file_path=str(dest_path),
                            content_type=content_type,
                            file_size=len(contents),
                            content_hash=content_hash,
                            extracted_data=combined_data,
                            embeddings=embeddings,
                            line_items=line_items,
                        )
                        detail = _document_detail_response(doc, extraction)
                        result = UploadFileResult(**detail.model_dump())
                if result.duplicate:
                    dest_path.unlink(missing_ok=True)
                    for page_path in page_paths:
                        page_path.unlink(missing_ok=True)
                return result
            except IntegrityError:
                write_db.rollback()
                with SessionLocal() as duplicate_db:
                    existing_doc = repo.get_document_by_content_hash(
                        duplicate_db,
                        content_hash,
                    )
                    if existing_doc:
                        latest = repo.get_latest_extraction(
                            duplicate_db,
                            existing_doc.id,
                        )
                        detail = _document_detail_response(
                            existing_doc,
                            latest,
                            duplicate=True,
                        )
                        dest_path.unlink(missing_ok=True)
                        for page_path in page_paths:
                            page_path.unlink(missing_ok=True)
                        return UploadFileResult(**detail.model_dump())
                raise
    except Exception:
        dest_path.unlink(missing_ok=True)
        for page_path in page_paths:
            page_path.unlink(missing_ok=True)
        raise


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: int, db: Session = Depends(get_db)) -> DocumentDetailResponse:
    doc = repo.get_document_with_latest_extraction(db=db, document_id=document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    latest = repo.get_latest_extraction(db=db, document_id=document_id)
    return _document_detail_response(doc, latest)


def _document_detail_response(
    doc: Document,
    latest: ExtractionResult | None,
    *,
    duplicate: bool = False,
) -> DocumentDetailResponse:
    extraction_response = None
    extracted_data = latest.extracted_data if latest and latest.extracted_data else None
    if extracted_data:
        extraction_response = _extraction_response(
            extracted_data,
            latest.model_name,
            latest.confidence_score,
        )
    return DocumentDetailResponse(
        id=str(doc.id),
        file_name=doc.file_name,
        status=doc.status,
        duplicate=duplicate,
        needs_review=extracted_data.get("needs_review", False) if extracted_data else False,
        file_path=doc.file_path,
        content_type=doc.content_type,
        file_size=doc.file_size,
        page_count=extracted_data.get("page_count", 1) if extracted_data else 1,
        pages=_pages_response(extracted_data.get("pages", []) if extracted_data else []),
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
