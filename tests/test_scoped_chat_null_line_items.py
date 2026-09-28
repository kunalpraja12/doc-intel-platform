from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.api.routes import chat
from db.models.document import Document
from db.models.extraction import ExtractionResult
from db.models.line_item import LineItem


def test_scoped_sql_chat_handles_null_document_and_line_item_fields():
    engine = create_engine("sqlite:///:memory:")
    for model in (Document, ExtractionResult, LineItem):
        model.__table__.create(engine)

    with Session(engine) as db:
        document = Document(file_name="null-values.pdf", status="completed")
        db.add(document)
        db.flush()
        db.add(
            ExtractionResult(
                document_id=document.id,
                extracted_data={
                    "fields": {
                        "vendor_name": None,
                        "document_type": "invoice",
                        "invoice_or_receipt_number": None,
                        "total_amount": None,
                    }
                },
            )
        )
        db.add(
            LineItem(
                document_id=document.id,
                description=None,
                quantity=None,
                unit_price=None,
                total=None,
                raw=None,
            )
        )
        db.commit()

        captured_context: list[str] = []

        def capture_context(question: str, contexts: list[str]) -> str:
            captured_context.extend(contexts)
            return "The invoice has one item, but its quantity and price are not available."

        with patch.object(chat, "_answer_with_context", side_effect=capture_context):
            response = chat.query_chat(
                "List all items with quantity and price.",
                [document.id],
                db,
            )

        assert response.answer.startswith("The invoice has one item")
        assert [source.document_id for source in response.sources] == [
            str(document.id)
        ]
        assert any("total=not available" in context for context in captured_context)
        item_context = next(
            context for context in captured_context if "items=" in context
        )
        assert item_context.count("'not available'") >= 3

        overview = chat.repo.get_structured_chat_overview(db, [document.id])
        assert overview["document_count"] == 1
        assert overview["quantity_total"] is None
        assert overview["line_net_total"] is None
