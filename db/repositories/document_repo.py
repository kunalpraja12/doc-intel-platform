"""Repository for document persistence.

Provides small helper methods used by the API routes to create documents,
persist extraction results, and query document+extraction state.
"""
from datetime import datetime
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select, desc

import db.models  # noqa: F401
from db.models.document import Document
from db.models.extraction import ExtractionResult


class DocumentRepository:
    """Repository with simple CRUD helpers for Document and ExtractionResult."""

    def __init__(self) -> None:
        self.name = "DocumentRepository"

    def create_document(
        self,
        db: Session,
        file_name: str,
        file_path: str,
        content_type: Optional[str] = None,
        file_size: Optional[int] = None,
        document_type: Optional[str] = None,
        uploaded_by: Optional[str] = None,
        status: str = "pending",
    ) -> Document:
        doc = Document(
            file_name=file_name,
            file_path=file_path,
            content_type=content_type,
            file_size=file_size,
            document_type=document_type,
            uploaded_by=uploaded_by,
            status=status,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    def update_status(self, db: Session, document_id: int, status: str) -> None:
        doc = db.get(Document, document_id)
        if not doc:
            return
        doc.status = status
        if status == "completed":
            doc.processed_at = datetime.utcnow()
        db.add(doc)
        db.commit()

    def add_extraction_result(
        self,
        db: Session,
        document_id: int,
        extracted_data: Dict[str, Any],
        confidence_score: Optional[float] = None,
        model_name: Optional[str] = None,
    ) -> ExtractionResult:
        res = ExtractionResult(
            document_id=document_id,
            extracted_data=extracted_data,
            confidence_score=confidence_score,
            model_name=model_name,
        )
        db.add(res)
        db.commit()
        db.refresh(res)
        return res

    def get_document_with_latest_extraction(self, db: Session, document_id: int) -> Optional[Document]:
        stmt = select(Document).where(Document.id == document_id)
        doc = db.execute(stmt).scalars().first()
        if not doc:
            return None
        # Accessing extraction_results will load relationships; the caller can inspect
        # doc.extraction_results to find the latest result.
        return doc

    def get_latest_extraction(self, db: Session, document_id: int) -> Optional[ExtractionResult]:
        stmt = select(ExtractionResult).where(ExtractionResult.document_id == document_id).order_by(desc(ExtractionResult.created_at))
        res = db.execute(stmt).scalars().first()
        return res
