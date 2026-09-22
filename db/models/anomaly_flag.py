"""Anomaly flags produced by validation/enrichment stages.

Records domain-specific issues (missing fields, out-of-range amounts, suspicious
vendors, duplicate invoices) for human review or automated workflows.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Boolean
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from db.session import Base


class AnomalyFlag(Base):
    __tablename__ = "anomaly_flags"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    field = Column(String, nullable=True)
    severity = Column(String, nullable=True)
    message = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="anomalies")
