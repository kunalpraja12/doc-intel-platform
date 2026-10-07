from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.schemas.document import StructuredLineItemResponse
from db.models.document import Document
from db.models.line_item import LineItem
from db.repositories.document_repo import DocumentRepository
from pipeline.extract.field_extraction import ExtractedLineItem
from pipeline.extract.field_extraction import (
    ExtractedDocumentFields,
    _ocr_row_percentages,
    supplement_line_items_from_ocr,
)


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


def test_ocr_supplements_missing_gst_percent_and_derives_exclusive_taxable_value():
    fields = ExtractedDocumentFields(
        taxable_amount_column_present=None,
        line_items=[
            ExtractedLineItem(
                description="MASOOR DAL SMALL 30 KG BAG",
                quantity=30,
                unit_price=104,
                base_rate=104,
                net_amount=3120,
                total=3120,
            ),
            ExtractedLineItem(
                description="BLACK SARSON 500 GM PACKET",
                quantity=4,
                unit_price=100,
                base_rate=95.24,
                net_amount=380.96,
                total=380.96,
            ),
            ExtractedLineItem(
                description="DHANIA GOTA 1 KG PACKET",
                quantity=3,
                unit_price=160,
                base_rate=152.38,
                net_amount=457.14,
                total=457.14,
            ),
            ExtractedLineItem(
                description="L BOW PASTA LOOSE",
                quantity=5,
                unit_price=52,
                base_rate=49.52,
                net_amount=247.60,
                total=247.60,
            ),
            ExtractedLineItem(
                description="MIRCHA POWDER PACKET",
                quantity=2,
                unit_price=280,
                base_rate=266.67,
                net_amount=533.34,
                total=533.34,
            ),
            ExtractedLineItem(
                description="HALDI POWDER PACKET",
                quantity=3,
                unit_price=190,
                base_rate=180.95,
                net_amount=542.85,
                total=542.85,
            ),
        ],
    )
    words = [
        {"text": "MASOOR", "left": 100, "top": 100, "width": 100, "height": 30},
        {"text": "DAL", "left": 210, "top": 100, "width": 50, "height": 30},
        {"text": "SMALL", "left": 270, "top": 100, "width": 100, "height": 30},
        {"text": "104.00", "left": 1000, "top": 100, "width": 100, "height": 30},
        {"text": "3120.00", "left": 2000, "top": 100, "width": 100, "height": 30},
        {"text": "BLACK", "left": 100, "top": 200, "width": 100, "height": 30},
        {"text": "SARSON", "left": 210, "top": 200, "width": 100, "height": 30},
        {"text": "95.24/5", "left": 1200, "top": 200, "width": 120, "height": 30},
        {"text": "%", "left": 1325, "top": 200, "width": 20, "height": 30},
        {"text": "380.96", "left": 2000, "top": 200, "width": 100, "height": 30},
        {"text": "DHANIA", "left": 100, "top": 300, "width": 100, "height": 30},
        {"text": "GOTA", "left": 210, "top": 300, "width": 80, "height": 30},
        {"text": "152385%", "left": 1200, "top": 300, "width": 120, "height": 30},
        {"text": "BOW", "left": 100, "top": 400, "width": 100, "height": 30},
        {"text": "PASTA", "left": 210, "top": 400, "width": 100, "height": 30},
        {"text": "49.525", "left": 1200, "top": 400, "width": 100, "height": 30},
        {"text": "%", "left": 1300, "top": 400, "width": 20, "height": 30},
        {"text": "MIRCHA", "left": 100, "top": 500, "width": 100, "height": 30},
        {"text": "POWDER", "left": 210, "top": 500, "width": 100, "height": 30},
        {"text": "KG,266.67.5", "left": 1200, "top": 500, "width": 150, "height": 30},
        {"text": "%", "left": 1350, "top": 500, "width": 20, "height": 30},
        {"text": "HALDI", "left": 100, "top": 600, "width": 100, "height": 30},
        {"text": "POWDER", "left": 210, "top": 600, "width": 100, "height": 30},
        {"text": "KG!180.9515", "left": 1200, "top": 600, "width": 150, "height": 30},
        {"text": "%", "left": 1350, "top": 600, "width": 20, "height": 30},
    ]

    enriched = supplement_line_items_from_ocr(fields, words)

    assert enriched.line_items[0].gst_percent is None
    assert enriched.line_items[0].taxable_amount is None
    assert enriched.line_items[1].gst_percent == 5
    assert enriched.line_items[1].taxable_amount == 380.96
    assert enriched.line_items[2].gst_percent == 5
    assert enriched.line_items[2].taxable_amount == 457.14
    assert enriched.line_items[3].gst_percent == 5
    assert enriched.line_items[3].taxable_amount == 247.60
    assert enriched.line_items[4].gst_percent == 5
    assert enriched.line_items[4].taxable_amount == 533.34
    assert enriched.line_items[5].gst_percent == 5
    assert enriched.line_items[5].taxable_amount == 542.85


def test_ocr_ignores_percentages_outside_the_gst_column():
    row = [
        {"text": "95.24)5%", "left": 1000, "top": 100, "width": 120, "height": 30},
        {"text": "28%", "left": 700, "top": 100, "width": 40, "height": 30},
    ]

    assert _ocr_row_percentages(row, base_rate=95.24, tax_column_x=1110) == {5}
