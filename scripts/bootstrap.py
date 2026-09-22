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
from sqlalchemy import inspect
from sqlalchemy.exc import SQLAlchemyError

from db.session import engine, SessionLocal, Base
from db.models.document import Document


def wait_for_db(timeout: int = 60, interval: int = 2) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            print("Postgres is available")
            return
        except SQLAlchemyError as exc:
            if time.time() > deadline:
                print(f"Timed out waiting for Postgres after {timeout}s: {exc}")
                raise
            print("Waiting for Postgres to be available...")
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
    session = SessionLocal()
    try:
        # Check if a sample row already exists
        existing = session.query(Document).filter(Document.uploaded_by == "bootstrap").first()
        if existing:
            print(f"Sample document already exists (id={existing.id})")
            return

        doc = Document(
            file_name="sample_invoice.pdf",
            document_type="invoice",
            status="uploaded",
            uploaded_by="bootstrap",
            metadata={"seed": True},
        )
        session.add(doc)
        session.commit()
        print(f"Inserted sample document with id={doc.id}")
    except Exception as exc:
        session.rollback()
        print(f"Failed to insert sample document: {exc}")
        raise
    finally:
        session.close()


def main() -> None:
    try:
        wait_for_db()
        ensure_tables()
        seed_sample_document()
        print("Bootstrap complete")
    except Exception as exc:
        print(f"Bootstrap failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
