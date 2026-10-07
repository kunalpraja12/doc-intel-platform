"""Line item model for invoice/receipt documents.

Each LineItem is associated to a Document and captures parsed item rows
from tabular sections (description, quantity, price, total) and preserved
raw OCR row data for reprocessing.
"""

from sqlalchemy import Column, Integer, Text, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship

from db.session import Base


class LineItem(Base):
    __tablename__ = "line_items"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    description = Column(Text, nullable=True)
    quantity = Column(Float, nullable=True)
    unit_price = Column(Float, nullable=True)
    gst_percent = Column(Float, nullable=True)
    total = Column(Float, nullable=True)
    raw = Column(JSON, nullable=True)  # raw OCR row or tokens

    document = relationship("Document", back_populates="line_items")
