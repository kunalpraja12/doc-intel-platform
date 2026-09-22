"""Health and readiness endpoints.

Includes a DB connectivity check using the shared get_db dependency.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.dependencies import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health(db: Session = Depends(get_db)) -> dict:
    try:
        # Simple DB query to verify connectivity
        db.execute(text("SELECT 1"))
        return {"status": "ok", "db": True}
    except Exception as exc:
        return {"status": "error", "db": False, "detail": str(exc)}
