"""Bootstrap script for local development.

- Waits for Postgres to become available
- (Optionally) creates tables using SQLAlchemy metadata if missing (convenience for dev)
- Inserts a single sample document row for quick testing

Usage:
  python scripts/bootstrap.py

This script is intended for local dev only.
"""

import time
import sys
from pathlib import Path
from typing import Optional
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.engine import make_url

# Ensure project root is on sys.path so imports like 'db' resolve when running
# this script directly (python scripts/bootstrap.py or python -m scripts.bootstrap)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from db.session import engine, SessionLocal, Base, DATABASE_URL
# Import all models to ensure SQLAlchemy mappers are configured for relationships
import db.models  # noqa: F401
from db.models.document import Document


def _mask_db_url(url: str) -> str:
    try:
        u = make_url(url)
        if u.password:
            u = u.set(password="****")
        return str(u)
    except Exception:
        return "<invalid-url>"


def wait_for_db(max_retries: int = 10, interval: int = 5) -> None:
    """Wait for the application's DATABASE_URL to be available.

    Uses the engine from db.session so it follows the same connection logic
    (e.g., a cloud Neon connection string). Prints detailed errors on failure
    and stops after max_retries.
    """
    print(f"Using DATABASE_URL: {_mask_db_url(DATABASE_URL)}")

    attempt = 0
    while attempt < max_retries:
        attempt += 1
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("Postgres is available")
            return
        except Exception as exc:
            # Print the exception so users can see why connection failed
            print(f"Attempt {attempt}/{max_retries} - DB connect failed: {exc!r}")
            if attempt >= max_retries:
                raise RuntimeError(f"Timed out waiting for DB after {max_retries} attempts: {exc}")
            time.sleep(interval)


def ensure_tables():
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    if "documents" not in tables:
        print("Documents table not found — creating tables from metadata (dev convenience)")
        Base.metadata.create_all(bind=engine)
    else:
        print("Required tables already exist")


def seed_sample_document():
    """Try to seed via ORM; if that fails due to schema mismatch, fall back to raw SQL.

    Falling back avoids bootstrap failing when the DB hasn't had a small follow-up
    migration applied yet (e.g., metadata_json column). The raw SQL path uses a
    simple INSERT ... RETURNING id which works as long as the documents table
    has the core columns (file_name, document_type, status, uploaded_by).
    """
    session = SessionLocal()
    try:
        # First try: ORM path (preferred)
        try:
            existing = session.query(Document).filter(Document.uploaded_by == "bootstrap").first()
        except Exception as orm_exc:
            # Likely a schema mismatch (missing column) — fall through to raw SQL
            print(f"ORM check failed (schema mismatch?), falling back to raw SQL: {orm_exc!r}")
            existing = None

        if existing:
            print(f"Sample document already exists (id={existing.id})")
            return

        # Try ORM insert if possible
        try:
            doc = Document(
                file_name="sample_invoice.pdf",
                document_type="invoice",
                status="uploaded",
                uploaded_by="bootstrap",
                metadata_json={"seed": True},
            )
            session.add(doc)
            session.commit()
            print(f"Inserted sample document with id={doc.id}")
            return
        except Exception as orm_insert_exc:
            session.rollback()
            print(f"ORM insert failed, falling back to raw SQL insert: {orm_insert_exc!r}")

        # Fallback: raw SQL insert using engine to avoid ORM schema expectations
        with engine.begin() as conn:
            # Check if a sample already exists (raw SQL)
            row = conn.execute(text("SELECT id FROM documents WHERE uploaded_by = :u LIMIT 1"), {"u": "bootstrap"}).fetchone()
            if row:
                print(f"Sample document already exists (id={row[0]})")
                return

            res = conn.execute(
                text(
                    "INSERT INTO documents (file_name, document_type, status, uploaded_by, created_at) "
                    "VALUES (:fn, :dt, :st, :ub, now()) RETURNING id"
                ),
                {"fn": "sample_invoice.pdf", "dt": "invoice", "st": "uploaded", "ub": "bootstrap"},
            )
            new_id = res.fetchone()[0]
            print(f"Inserted sample document with id={new_id} (raw SQL)")
            return

    except Exception as exc:
        session.rollback()
        print(f"Failed to insert sample document (both ORM and raw SQL failed): {exc!r}")
        raise
    finally:
        session.close()


def main() -> None:
    # Allow optional first arg to control max_retries when invoking the script
    max_retries = 10
    if len(sys.argv) > 1:
        try:
            max_retries = int(sys.argv[1])
        except Exception:
            print(f"Invalid max_retries value: {sys.argv[1]}, falling back to {max_retries}")

    try:
        wait_for_db(max_retries=max_retries)
        ensure_tables()
        seed_sample_document()
        print("Bootstrap complete")
    except Exception as exc:
        print(f"Bootstrap failed: {exc!r}")
        sys.exit(1)


if __name__ == "__main__":
    main()
