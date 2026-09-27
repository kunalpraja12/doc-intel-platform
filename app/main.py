"""FastAPI application entrypoint for the Doc Intel Platform.

Wires the DB lifespan check and registers routers.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import text, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from db.session import engine
from app.dependencies import get_db

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Verify DB connectivity on startup; fail loudly if it cannot connect.
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        raise RuntimeError(f"Database connection failed during startup: {exc}")

    yield


app = FastAPI(title="Doc Intel Platform", version="0.1.0", lifespan=lifespan)

# Register routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
