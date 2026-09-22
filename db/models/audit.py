"""Audit trail model placeholder.

TODO: Support evidence logging and human review flags for each extraction step.
"""

from sqlalchemy import Column, Integer, String, Text

from db.session import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False)
    action = Column(String, nullable=False)
    details = Column(Text, nullable=True)
