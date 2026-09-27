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
from io import BytesIO
import hashlib
import os
import logging
import uuid
import zipfile

from app.api.schemas.document import (
    DocumentResponse,
    DocumentDetailResponse,
    ExtractionResultResponse,
    StructuredFieldsResponse,
    StructuredLineItemResponse,
    PageExtractionResponse,
    UploadFileResult,
)
from app.dependencies import get_db
# Ensure all models are imported so relationships resolve
import db.models  # noqa: F401
from db.models.document import Document
from db.models.extraction import ExtractionResult
from db.repositories.document_repo import DocumentRepository

# Import OCR extractor
from pipeline.extract.ocr import OCRExtractor
from pipeline.extract.field_extraction import extract_fields
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
SUPPORTED_CONTENT_TYPES_MESSAGE = ", ".join(sorted(SUPPORTED_CONTENT_TYPES))
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


@router.post("/upload", response_model=list[UploadFileResult])
async def upload_document(
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> list[UploadFileResult]:
    results: list[UploadFileResult] = []
    for uploaded_file in files:
        filename = Path(uploaded_file.filename or "uploaded-file").name
        try:
            if filename.casefold().endswith(".zip"):
                archive_bytes = await uploaded_file.read()
                results.extend(_process_zip_archive(archive_bytes, filename, db))
                continue

            if uploaded_file.size is not None and uploaded_file.size > MAX_FILE_SIZE:
                results.append(_upload_error(filename, "File exceeds 20MB limit"))
                continue

            contents = await uploaded_file.read()
            if len(contents) > MAX_FILE_SIZE:
                results.append(_upload_error(filename, "File exceeds 20MB limit"))
                continue

            content_type = _content_type(filename, uploaded_file.content_type)
            if content_type not in SUPPORTED_CONTENT_TYPES:
                results.append(
                    _upload_error(
                        filename,
                        f"Unsupported file type. Supported types: {SUPPORTED_CONTENT_TYPES_MESSAGE}, ZIP",
                    )
                )
                continue
            results.append(_process_document_upload(contents, filename, content_type, db))
        except Exception as exc:
            logger.exception("Failed to process upload %s", filename)
            results.append(_upload_error(filename, str(exc)))
        finally:
            await uploaded_file.close()
    return results


def _upload_error(filename: str, message: str) -> UploadFileResult:
    return UploadFileResult(
        file_name=filename,
        status=(
            "skipped"
            if message.startswith(("Skipped", "File exceeds"))
            else "failed"
        ),
        error=message,
    )


def _content_type(filename: str, provided_type: str | None) -> str | None:
    return CONTENT_TYPE_BY_SUFFIX.get(Path(filename).suffix.casefold(), provided_type)


def _process_zip_archive(
    archive_bytes: bytes,
    archive_name: str,
    db: Session,
) -> list[UploadFileResult]:
    results: list[UploadFileResult] = []
    try:
        archive = zipfile.ZipFile(BytesIO(archive_bytes))
    except zipfile.BadZipFile:
        return [_upload_error(archive_name, "Invalid ZIP archive")]

    with archive:
        for entry in archive.infolist():
            entry_name = entry.filename
            normalized_name = entry_name.replace("\\", "/")
            parts = normalized_name.split("/")
            safe_name = Path(parts[-1]) if parts else Path(entry_name)
            display_name = f"{archive_name}:{entry_name}"

            if entry.is_dir():
                results.append(_upload_error(display_name, "Skipped folder entry"))
                continue
            if any(part.startswith(".") or part == "__MACOSX" for part in parts):
                results.append(_upload_error(display_name, "Skipped hidden or system file"))
                continue
            content_type = _content_type(safe_name.name, None)
            if content_type not in SUPPORTED_CONTENT_TYPES:
                results.append(_upload_error(display_name, "Skipped unsupported file type"))
                continue
            if entry.file_size > MAX_FILE_SIZE:
                results.append(_upload_error(display_name, "File exceeds 20MB limit"))
                continue
            try:
                contents = archive.read(entry)
                if len(contents) > MAX_FILE_SIZE:
                    results.append(_upload_error(display_name, "File exceeds 20MB limit"))
                    continue
                results.append(
                    _process_document_upload(contents, safe_name.name, content_type, db)
                )
            except Exception as exc:
                logger.exception("Failed to process ZIP entry %s", display_name)
                results.append(_upload_error(display_name, str(exc)))
    return results


def _process_document_upload(
    contents: bytes,
    filename: str,
    content_type: str,
    db: Session,
) -> UploadFileResult:
    safe_name = Path(filename).name
    content_hash = hashlib.sha256(contents).hexdigest()
    existing_doc = repo.get_document_by_content_hash(db, content_hash)
    if existing_doc:
        latest = repo.get_latest_extraction(db, existing_doc.id)
        detail = _document_detail_response(existing_doc, latest, duplicate=True)
        return UploadFileResult(**detail.model_dump())

    unique_prefix = uuid.uuid4().hex
    dest_path = UPLOAD_DIR / f"{unique_prefix}_{safe_name}"
    dest_path.write_bytes(contents)
    try:
        doc = repo.create_document(
            db=db,
            file_name=safe_name,
            file_path=str(dest_path),
            content_type=content_type,
            file_size=len(contents),
            status="pending",
            content_hash=content_hash,
        )
    except IntegrityError as exc:
        db.rollback()
        existing_doc = repo.get_document_by_content_hash(db, content_hash)
        if existing_doc:
            dest_path.unlink(missing_ok=True)
            latest = repo.get_latest_extraction(db, existing_doc.id)
            detail = _document_detail_response(existing_doc, latest, duplicate=True)
            return UploadFileResult(**detail.model_dump())
        raise RuntimeError("Failed to create document record") from exc

    try:
        if content_type == "application/pdf":
            page_paths = _render_pdf_pages(
                contents,
                UPLOAD_DIR,
                f"{unique_prefix}_{Path(safe_name).stem}",
            )
            extractor = OCRExtractor()
            page_texts: list[str] = []
            page_words: list[dict] = []
            for page_number, page_path in enumerate(page_paths, start=1):
                page_result = extractor.extract_from_path(str(page_path), use_preprocess=True)
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

        page_results = [
            {"page_number": number, "text": text, "fields": None}
            for number, text in enumerate(page_texts, start=1)
        ]
        raw_data = {
            "page_count": len(page_results),
            "pages": page_results,
            "text": extracted["text"],
            "words": extracted["words"],
        }
        extraction_result = repo.add_extraction_result(
            db=db,
            document_id=doc.id,
            extracted_data=raw_data,
            model_name="tesseract",
        )
        for page in page_results:
            page["fields"] = extract_fields(page["text"]).model_dump()

        combined_data = {
            **raw_data,
            "pages": page_results,
            "fields": page_results[0]["fields"],
        }
        repo.update_extraction_data(db, extraction_result.id, combined_data)
        for page in page_results:
            chunk = page_chunk_text(page["page_number"], page["text"], page["fields"])
            repo.add_embedding(
                db=db,
                document_id=doc.id,
                page_number=page["page_number"] if content_type == "application/pdf" else None,
                embedding=embed_text(chunk),
                chunk_text=chunk,
            )
        repo.add_line_items(
            db,
            doc.id,
            [
                item
                for page in page_results
                for item in page["fields"].get("line_items", [])
            ],
        )
        repo.update_status(db, doc.id, "completed")
        detail = _document_detail_response(
            doc,
            repo.get_latest_extraction(db, doc.id),
        )
        return UploadFileResult(**detail.model_dump())
    except Exception:
        repo.update_status(db, doc.id, "failed")
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
