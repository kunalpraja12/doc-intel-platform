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


def test_named_invoice_with_null_total_uses_only_its_line_item_net_amounts():
    engine = create_engine("sqlite:///:memory:")
    for model in (Document, ExtractionResult, LineItem):
        model.__table__.create(engine)

    with Session(engine) as db:
        target = Document(file_name="SSEPL-67999.pdf", status="completed")
        unrelated = Document(file_name="unrelated-invoice.pdf", status="completed")
        db.add_all([target, unrelated])
        db.flush()
        db.add_all(
            [
                ExtractionResult(
                    document_id=target.id,
                    extracted_data={
                        "fields": {
                            "vendor_name": "SSEPL",
                            "document_type": "invoice",
                            "invoice_or_receipt_number": "SSEPL-67999",
                            "total_amount": None,
                        }
                    },
                ),
                ExtractionResult(
                    document_id=unrelated.id,
                    extracted_data={
                        "fields": {
                            "vendor_name": "Other vendor",
                            "document_type": "invoice",
                            "invoice_or_receipt_number": "OTHER-1",
                            "total_amount": 59573.00,
                        }
                    },
                ),
            ]
        )
        db.add_all(
            [
                LineItem(
                    document_id=target.id,
                    description=f"Target item {index}",
                    total=amount,
                    raw={"net_amount": amount},
                )
                for index, amount in enumerate(
                    [10000.00, 10000.00, 10000.00, 10000.00, 435.80]
                )
            ]
            + [
                LineItem(
                    document_id=unrelated.id,
                    description="Unrelated item",
                    total=59573.00,
                    raw={"net_amount": 59573.00},
                )
            ]
        )
        db.commit()

        with patch.object(
            chat,
            "_answer_with_context",
            side_effect=AssertionError("Null-total answer must not use Gemini"),
        ):
            response = chat.query_chat(
                "What is the total of invoice SSEPL-67999",
                None,
                db,
            )

        assert "wasn't identified" in response.answer.casefold()
        assert "₹40,435.80" in response.answer
        assert "59,573.00" not in response.answer
        assert [source.document_id for source in response.sources] == [
            str(target.id)
        ]


def test_any_named_document_with_null_total_uses_its_own_line_item_sum():
    engine = create_engine("sqlite:///:memory:")
    for model in (Document, ExtractionResult, LineItem):
        model.__table__.create(engine)

    with Session(engine) as db:
        document = Document(file_name="ACME-2026-04.pdf", status="completed")
        db.add(document)
        db.flush()
        db.add(
            ExtractionResult(
                document_id=document.id,
                extracted_data={
                    "fields": {
                        "document_type": "invoice",
                        "invoice_or_receipt_number": "ACME-2026-04",
                        "total_amount": None,
                    }
                },
            )
        )
        db.add(
            LineItem(
                document_id=document.id,
                description="Captured item",
                total=120.50,
                raw={"net_amount": 120.50},
            )
        )
        db.commit()

        with patch.object(
            chat,
            "_answer_with_context",
            side_effect=AssertionError("Null-total answer must not use Gemini"),
        ):
            response = chat.query_chat(
                "What is the total of invoice ACME-2026-04",
                None,
                db,
            )

        assert "wasn't identified" in response.answer.casefold()
        assert "₹120.50" in response.answer
        assert [source.document_id for source in response.sources] == [
            str(document.id)
        ]
