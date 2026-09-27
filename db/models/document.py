"""Document model with basic metadata and relationships.

Minimal starter schema for documents used by the extraction pipeline.
"""

from sqlalchemy import Column, String, DateTime, Integer, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from db.session import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=True)
    content_hash = Column(String(64), nullable=True, unique=True, index=True)
    content_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    document_type = Column(String, nullable=True)
    status = Column(String, default="uploaded", nullable=False)
    uploaded_by = Column(String, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    extraction_results = relationship(
        "ExtractionResult", back_populates="document", cascade="all, delete-orphan"
    )
    line_items = relationship("LineItem", back_populates="document", cascade="all, delete-orphan")
    anomalies = relationship("AnomalyFlag", back_populates="document", cascade="all, delete-orphan")
