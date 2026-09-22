"""Database session and engine configuration.

Loads DATABASE_URL from environment (.env) and configures SQLAlchemy engine
and session factory. Falls back to POSTGRES_* env vars or a sensible local
default. Normalizes the PostgreSQL URL scheme to use psycopg3 when needed.
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# Load environment from .env file when present (local dev convenience)
load_dotenv()

# Primary: explicit DATABASE_URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Secondary: construct from POSTGRES_* env vars if DATABASE_URL is not provided
if not DATABASE_URL:
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "docintel")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    DATABASE_URL = (
        f"postgresql+psycopg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )

# Normalize scheme so SQLAlchemy picks the psycopg (psycopg3) driver when available.
# If the user provided "postgresql://..." or "postgres://..." in .env, convert it to
# "postgresql+psycopg://..." which points SQLAlchemy at psycopg3 (installed as psycopg).
if DATABASE_URL and DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
elif DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)

Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
