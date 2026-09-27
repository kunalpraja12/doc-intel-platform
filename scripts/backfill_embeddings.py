"""Backfill page-level Gemini embeddings for existing documents.

Run from the project root with:
    python -m scripts.backfill_embeddings
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from db.models.document import Document  # noqa: E402
from db.repositories.document_repo import DocumentRepository  # noqa: E402
from db.session import SessionLocal  # noqa: E402
from pipeline.extract.embeddings import embed_text, page_chunk_text  # noqa: E402


def backfill() -> None:
    repo = DocumentRepository()
    with SessionLocal() as db:
        documents = db.query(Document).order_by(Document.id).all()
        for document in documents:
            extraction = repo.get_latest_extraction(db, document.id)
            if not extraction or not extraction.extracted_data:
                print(f"Skipping document {document.id}: no extraction data")
                continue

            data = extraction.extracted_data
            pages = data.get("pages") or [
                {
                    "page_number": 1,
                    "text": data.get("text", ""),
                    "fields": data.get("fields") or {},
                }
            ]
            repo.delete_embeddings(db, document.id)
            for page in pages:
                chunk = page_chunk_text(
                    page["page_number"],
                    page.get("text", ""),
                    page.get("fields") or {},
                )
                repo.add_embedding(
                    db=db,
                    document_id=document.id,
                    page_number=(
                        page["page_number"]
                        if data.get("page_count", 1) > 1
                        else None
                    ),
                    embedding=embed_text(chunk),
                    chunk_text=chunk,
                )
            print(f"Backfilled document {document.id}: {len(pages)} page embedding(s)")


if __name__ == "__main__":
    backfill()
