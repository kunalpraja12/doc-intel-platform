"""Document upload and metadata routes.

Lightweight listing endpoint to validate API -> DB round trip.
"""

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas.document import DocumentResponse
from app.dependencies import get_db
# Ensure all models are imported so relationships resolve
import db.models  # noqa: F401
from db.models.document import Document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentResponse])
async def list_documents(db: Session = Depends(get_db)) -> list[DocumentResponse]:
    stmt = select(Document)
    results = db.execute(stmt).scalars().all()
    return [DocumentResponse(id=str(d.id), file_name=d.file_name, status=d.status) for d in results]
