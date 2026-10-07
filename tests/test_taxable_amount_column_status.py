from app.api.schemas.document import StructuredFieldsResponse
from pipeline.extract.field_extraction import ExtractedDocumentFields


def test_taxable_amount_column_presence_is_carried_to_upload_response():
    extracted = ExtractedDocumentFields(
        taxable_amount_column_present=False,
        line_items=[{"description": "Item", "taxable_amount": None}],
    )

    response = StructuredFieldsResponse.model_validate(extracted.model_dump())

    assert response.taxable_amount_column_present is False
    assert response.line_items[0].taxable_amount is None


def test_unknown_taxable_column_status_remains_distinct_from_absent():
    response = StructuredFieldsResponse(
        taxable_amount_column_present=None,
        line_items=[{"description": "Item", "taxable_amount": None}],
    )

    assert response.taxable_amount_column_present is None
