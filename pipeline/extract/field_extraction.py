"""LLM-assisted extraction of structured document fields from OCR text."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()


class ExtractedLineItem(BaseModel):
    """A best-effort line item parsed from imperfect OCR text."""

    description: str | None = Field(default=None)
    quantity: float | None = Field(default=None)
    unit_price: float | None = Field(default=None)
    total: float | None = Field(default=None)


class ExtractedDocumentFields(BaseModel):
    """Structured fields extracted from a receipt, invoice, or other document."""

    vendor_name: str | None = Field(default=None)
    document_type: str | None = Field(
        default=None,
        description="Best guess: invoice, receipt, or other.",
    )
    total_amount: float | None = Field(default=None)
    date: str | None = Field(default=None, description="Date as printed in the document.")
    invoice_or_receipt_number: str | None = Field(default=None)
    line_items: list[ExtractedLineItem] = Field(default_factory=list)


def extract_fields(
    raw_ocr_text: str,
    *,
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
) -> ExtractedDocumentFields:
    """Extract structured fields from raw OCR text using Gemini structured output.

    The model is instructed to preserve uncertainty as null rather than inventing
    values. A missing API key is reported explicitly instead of returning a
    success-shaped empty result.
    """

    if not raw_ocr_text.strip():
        return ExtractedDocumentFields()

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for LLM field extraction")

    llm = ChatGoogleGenerativeAI(
        model=model,
        google_api_key=resolved_api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ExtractedDocumentFields)
    prompt = """Extract fields from the OCR text below.

The OCR may contain spelling errors, missing punctuation, merged words, and
columns in the wrong order. Do your best to identify values using the context
of the entire document. Return null for any field that cannot be determined
with reasonable confidence; do not hallucinate or infer unsupported values.
For line items, include only rows that are reasonably identifiable. Use the
document's printed date format for `date`, and classify `document_type` as
exactly `invoice`, `receipt`, or `other` when possible.

OCR text:
---
""" + raw_ocr_text + """
---
"""
    result = structured_llm.invoke(prompt)
    if isinstance(result, ExtractedDocumentFields):
        return result
    return ExtractedDocumentFields.model_validate(result)
