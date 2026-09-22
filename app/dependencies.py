"""Shared dependency providers for request handling.

Provides a get_db dependency that yields a SQLAlchemy session per request
and ensures it is closed after the request completes.
"""

from typing import Generator

from db.session import SessionLocal


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
