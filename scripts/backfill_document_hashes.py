"""Populate SHA-256 hashes for previously uploaded document files.

Run after applying migration 0004:
    python -m scripts.backfill_document_hashes
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from db.models.document import Document  # noqa: E402
from db.session import SessionLocal  # noqa: E402


def backfill_hashes() -> None:
    with SessionLocal() as db:
        documents = db.query(Document).order_by(Document.id).all()
        known_hashes = {
            document.content_hash
            for document in documents
            if document.content_hash is not None
        }

        for document in documents:
            if document.content_hash is not None:
                continue
            if not document.file_path:
                print(f"Skipping document {document.id}: no file path")
                continue

            file_path = Path(document.file_path)
            if not file_path.is_file():
                print(f"Skipping document {document.id}: file is unavailable at {file_path}")
                continue

            content_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            if content_hash in known_hashes:
                print(
                    f"Skipping document {document.id}: identical content hash "
                    "is already assigned to another document"
                )
                continue

            document.content_hash = content_hash
            known_hashes.add(content_hash)
            db.commit()
            print(f"Hashed document {document.id}")


if __name__ == "__main__":
    backfill_hashes()
