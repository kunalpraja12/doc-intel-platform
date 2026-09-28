"""Health and readiness endpoints.

Includes a DB connectivity check using the shared get_db dependency.
"""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.dependencies import get_db

router = APIRouter(prefix="/health", tags=["health"])
logger = logging.getLogger(__name__)


@router.get("")
async def health(db: Session = Depends(get_db)) -> dict:
    try:
        # Simple DB query to verify connectivity
        db.execute(text("SELECT 1"))
        return {"status": "ok", "db": True}
    except Exception:
        logger.exception("Health check database probe failed")
        return {"status": "error", "db": False}
