"""FastAPI application entrypoint for the Doc Intel Platform.

Wires the DB lifespan check and registers routers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text

from app.core.errors import (
    AI_BUSY_MESSAGE,
    AI_QUOTA_MESSAGE,
    DATABASE_MESSAGE,
    FILE_MESSAGE,
    FILE_TOO_LARGE_MESSAGE,
    GENERIC_MESSAGE,
    is_friendly_message,
    user_friendly_message,
)
from db.session import engine
from app.dependencies import get_db

# Import routers
from app.api.routes.health import router as health_router
from app.api.routes.documents import router as documents_router
from app.api.routes.chat import router as chat_router

logger = logging.getLogger(__name__)


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


@app.exception_handler(StarletteHTTPException)
async def safe_http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    logger.warning(
        "HTTP request failed with status %s: %r",
        exc.status_code,
        exc.detail,
    )
    detail = exc.detail
    if is_friendly_message(detail):
        message = detail
    elif exc.status_code == 413 and request.url.path == "/documents/upload":
        message = FILE_TOO_LARGE_MESSAGE
    elif exc.status_code == 415 and request.url.path == "/documents/upload":
        message = FILE_MESSAGE
    elif exc.status_code == 429:
        message = AI_QUOTA_MESSAGE
    elif exc.status_code == 503:
        message = (
            DATABASE_MESSAGE
            if "database" in str(detail).casefold()
            else AI_BUSY_MESSAGE
        )
    else:
        message = GENERIC_MESSAGE
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": message},
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def safe_validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    logger.warning("Request validation failed: %r", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"message": GENERIC_MESSAGE},
    )


@app.exception_handler(Exception)
async def safe_unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error(
        "Unhandled request exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={"message": user_friendly_message(exc)},
    )


# Register routers
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)


@app.get("/", include_in_schema=False)
async def frontend() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
