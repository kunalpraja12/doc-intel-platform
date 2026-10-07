from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.schemas.document import StructuredLineItemResponse
from db.models.document import Document
from db.models.line_item import LineItem
from db.repositories.document_repo import DocumentRepository
from pipeline.extract.field_extraction import ExtractedLineItem


def test_gst_percent_is_extracted_and_exposed_as_line_item_field():
    extracted = ExtractedLineItem(
        description="Spice packet",
        gst_percent=5,
        taxable_amount=100,
    )

    response = StructuredLineItemResponse.model_validate(extracted.model_dump())

    assert response.gst_percent == 5


def test_gst_percent_is_persisted_in_line_item_column_and_raw_data():
    engine = create_engine("sqlite:///:memory:")
    Document.__table__.create(engine)
    LineItem.__table__.create(engine)

    with Session(engine) as db:
        document = Document(file_name="gst-invoice.jpg", status="completed")
        db.add(document)
        db.flush()

        line = ExtractedLineItem(
            description="Spice packet",
            quantity=2,
            unit_price=50,
            gst_percent=5,
            total=95,
        )
        DocumentRepository().add_line_items(
            db,
            document.id,
            [line.model_dump()],
        )

        persisted = db.query(LineItem).filter_by(document_id=document.id).one()
        assert persisted.gst_percent == 5
        assert persisted.raw["gst_percent"] == 5
