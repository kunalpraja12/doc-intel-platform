"""Repository for document persistence.

Provides small helper methods used by the API routes to create documents,
persist extraction results, and query document+extraction state.
"""
from datetime import datetime, timezone
import re
from typing import Optional, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import desc, func, select

import db.models  # noqa: F401
from db.models.document import Document
from db.models.extraction import ExtractionResult
from db.models.line_item import LineItem
from db.models.embedding import DocumentEmbedding


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    normalized = re.sub(r"[^0-9.+-]", "", str(value).replace(",", ""))
    try:
        return float(normalized) if normalized else None
    except ValueError:
        return None


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
        content_hash: Optional[str] = None,
    ) -> Document:
        doc = Document(
            file_name=file_name,
            file_path=file_path,
            content_type=content_type,
            file_size=file_size,
            document_type=document_type,
            uploaded_by=uploaded_by,
            status=status,
            content_hash=content_hash,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc

    def get_document_by_content_hash(
        self,
        db: Session,
        content_hash: str,
    ) -> Optional[Document]:
        return (
            db.query(Document)
            .filter(Document.content_hash == content_hash)
            .first()
        )

    def persist_processed_upload(
        self,
        db: Session,
        *,
        file_name: str,
        file_path: str,
        content_type: str,
        file_size: int,
        content_hash: str,
        extracted_data: Dict[str, Any],
        embeddings: list[Dict[str, Any]],
        line_items: list[Dict[str, Any]],
    ) -> tuple[Document, ExtractionResult]:
        """Stage all already-computed upload data in the caller's transaction."""
        document = Document(
            file_name=file_name,
            file_path=file_path,
            content_type=content_type,
            file_size=file_size,
            content_hash=content_hash,
            status="completed",
            processed_at=datetime.now(timezone.utc),
        )
        extraction = ExtractionResult(
            document=document,
            extracted_data=extracted_data,
            model_name="tesseract+gemini",
        )
        db.add_all([document, extraction])
        db.flush()

        db.add_all([
            LineItem(
                document_id=document.id,
                description=item.get("description"),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total=item.get("total"),
                raw=item,
            )
            for item in line_items
        ])
        db.add_all([
            DocumentEmbedding(
                document_id=document.id,
                page_number=item["page_number"],
                embedding=item["embedding"],
                chunk_text=item["chunk_text"],
            )
            for item in embeddings
        ])
        db.flush()
        return document, extraction

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

    def update_extraction_data(
        self,
        db: Session,
        extraction_id: int,
        extracted_data: Dict[str, Any],
    ) -> ExtractionResult:
        result = db.get(ExtractionResult, extraction_id)
        if not result:
            raise ValueError(f"Extraction result {extraction_id} not found")
        result.extracted_data = extracted_data
        db.add(result)
        db.commit()
        db.refresh(result)
        return result

    def add_line_items(
        self,
        db: Session,
        document_id: int,
        line_items: list[Dict[str, Any]],
    ) -> list[LineItem]:
        rows = [
            LineItem(
                document_id=document_id,
                description=item.get("description"),
                quantity=item.get("quantity"),
                unit_price=item.get("unit_price"),
                total=item.get("total"),
                raw=item,
            )
            for item in line_items
        ]
        if rows:
            db.add_all(rows)
            db.commit()
            for row in rows:
                db.refresh(row)
        return rows

    def add_embedding(
        self,
        db: Session,
        document_id: int,
        page_number: Optional[int],
        embedding: list[float],
        chunk_text: str,
    ) -> DocumentEmbedding:
        row = DocumentEmbedding(
            document_id=document_id,
            page_number=page_number,
            embedding=embedding,
            chunk_text=chunk_text,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row

    def delete_embeddings(self, db: Session, document_id: int) -> None:
        db.query(DocumentEmbedding).filter(
            DocumentEmbedding.document_id == document_id
        ).delete(synchronize_session=False)
        db.commit()

    def search_embeddings(
        self,
        db: Session,
        query_embedding: list[float],
        limit: int = 5,
        document_ids: list[int] | None = None,
    ) -> list[tuple[DocumentEmbedding, float]]:
        if document_ids is not None and not document_ids:
            return []

        candidate_limit = max(limit * 4, 20)
        per_document_limit = 2
        distance = DocumentEmbedding.embedding.cosine_distance(query_embedding)
        query = db.query(DocumentEmbedding, distance.label("distance"))
        if document_ids is not None:
            query = query.filter(DocumentEmbedding.document_id.in_(document_ids))
        candidates = query.order_by(distance).limit(candidate_limit).all()

        selected: list[tuple[DocumentEmbedding, float]] = []
        document_counts: dict[int, int] = {}
        for embedding, distance_value in candidates:
            count = document_counts.get(embedding.document_id, 0)
            if count >= per_document_limit:
                continue
            selected.append((embedding, float(distance_value)))
            document_counts[embedding.document_id] = count + 1

        selected.sort(key=lambda item: item[1])
        return selected[:limit]

    def get_structured_chat_overview(
        self,
        db: Session,
        document_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        """Return compact document summaries and SQL aggregates without line-item rows."""
        if document_ids is not None and not document_ids:
            return {
                "document_count": 0,
                "invoice_count": 0,
                "receipt_count": 0,
                "line_item_count": 0,
                "quantity_total": 0.0,
                "line_net_total": 0.0,
                "document_total": 0.0,
                "documents": [],
            }

        latest_extractions = (
            db.query(
                ExtractionResult.document_id.label("document_id"),
                func.max(ExtractionResult.id).label("latest_id"),
            )
            .group_by(ExtractionResult.document_id)
            .subquery()
        )
        vendor_name = ExtractionResult.extracted_data["fields"]["vendor_name"].as_string()
        invoice_number = ExtractionResult.extracted_data["fields"][
            "invoice_or_receipt_number"
        ].as_string()
        extracted_type = ExtractionResult.extracted_data["fields"][
            "document_type"
        ].as_string()
        total_amount = ExtractionResult.extracted_data["fields"][
            "total_amount"
        ].as_string()
        document_query = (
            db.query(
                Document.id,
                Document.file_name,
                Document.document_type,
                vendor_name.label("vendor_name"),
                invoice_number.label("invoice_number"),
                extracted_type.label("extracted_type"),
                total_amount.label("total_amount"),
            )
            .outerjoin(
                latest_extractions,
                latest_extractions.c.document_id == Document.id,
            )
            .outerjoin(
                ExtractionResult,
                ExtractionResult.id == latest_extractions.c.latest_id,
            )
            .order_by(Document.id)
        )
        if document_ids is not None:
            document_query = document_query.filter(Document.id.in_(document_ids))

        document_rows = document_query.all()
        line_stats_query = (
            db.query(
                func.count(LineItem.id),
                func.count(LineItem.quantity),
                func.sum(LineItem.quantity),
                func.count(LineItem.total),
                func.sum(LineItem.total),
            )
            .join(Document, Document.id == LineItem.document_id)
        )
        if document_ids is not None:
            line_stats_query = line_stats_query.filter(
                LineItem.document_id.in_(document_ids)
            )
        (
            line_item_count,
            quantity_count,
            quantity_total,
            net_amount_count,
            line_net_total,
        ) = line_stats_query.one()

        documents: list[dict[str, Any]] = []
        invoice_count = 0
        receipt_count = 0
        document_total = 0.0
        document_total_count = 0
        for row in document_rows:
            document_type = row.document_type or row.extracted_type
            normalized_type = (document_type or "").casefold()
            invoice_count += normalized_type == "invoice"
            receipt_count += normalized_type == "receipt"
            parsed_total = _optional_float(row.total_amount)
            if parsed_total is not None:
                document_total += parsed_total
                document_total_count += 1
            documents.append(
                {
                    "document_id": row.id,
                    "file_name": row.file_name,
                    "vendor_name": row.vendor_name,
                    "invoice_number": row.invoice_number,
                    "document_type": document_type,
                    "total_amount": parsed_total,
                }
            )

        return {
            "document_count": len(document_rows),
            "invoice_count": invoice_count,
            "receipt_count": receipt_count,
            "line_item_count": int(line_item_count or 0),
            "quantity_total": (
                float(quantity_total) if quantity_count else None
            ),
            "line_net_total": (
                float(line_net_total) if net_amount_count else None
            ),
            "document_total": document_total if document_total_count else None,
            "documents": documents,
        }

    def get_line_items_for_documents(
        self,
        db: Session,
        document_ids: list[int],
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Load a bounded line-item sample for a limited document subset."""
        if not document_ids or limit <= 0:
            return []
        rows = (
            db.query(
                LineItem.document_id,
                LineItem.description,
                LineItem.quantity,
                LineItem.unit_price,
                LineItem.total,
            )
            .filter(LineItem.document_id.in_(document_ids))
            .order_by(LineItem.document_id, LineItem.id)
            .limit(limit)
            .all()
        )
        return [
            {
                "document_id": row.document_id,
                "description": row.description,
                "quantity": row.quantity,
                "unit_price": row.unit_price,
                "net_amount": row.total,
            }
            for row in rows
        ]
