"""Extraction results for a processed document.

Stores raw/normalized extraction output, confidence and model metadata.
"""

from sqlalchemy import Column, Integer, JSON, String, ForeignKey, DateTime, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from db.session import Base


class ExtractionResult(Base):
    __tablename__ = "extraction_results"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(
        Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    extracted_data = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    model_name = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="extraction_results")
