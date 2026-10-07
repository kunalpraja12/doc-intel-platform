"""LLM-assisted extraction of structured document fields from OCR text."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.core.errors import invoke_gemini_with_retries

load_dotenv()

_STANDARD_GST_RATES = {0, 0.1, 0.25, 1, 1.5, 3, 5, 6, 7.5, 12, 18, 28, 40}


class ExtractedLineItem(BaseModel):
    """A best-effort line item parsed from imperfect OCR text."""

    description: str | None = Field(default=None)
    quantity: float | None = Field(default=None)
    cases: float | None = Field(default=None, description="Value in a Cs/cases column, when present.")
    pieces: float | None = Field(default=None, description="Value in a Pcs column, when present.")
    units_per_case: float | None = Field(default=None, description="UPC value, when present.")
    base_rate: float | None = Field(default=None, description="Base rate used to calculate the taxable amount.")
    unit_price: float | None = Field(default=None)
    gst_percent: float | None = Field(
        default=None,
        description=(
            "Per-line GST/tax percentage shown in a GST, GST @, Tax, or Tax @ "
            "column. Return the numeric percentage, e.g. 5 for 5%; null if blank."
        ),
    )
    discount: float | None = Field(default=None)
    taxable_amount: float | None = Field(default=None)
    net_amount: float | None = Field(default=None)
    total: float | None = Field(default=None)
    needs_review: bool = Field(default=False)


class ExtractedDocumentFields(BaseModel):
    """Structured fields extracted from a receipt, invoice, or other document."""

    vendor_name: str | None = Field(default=None)
    taxable_amount_column_present: bool | None = Field(
        default=None,
        description=(
            "Whether the source page has a separate taxable amount/value column "
            "for line items. Do not count a per-unit rate excluding GST as a "
            "separate taxable amount column."
        ),
    )
    document_type: str | None = Field(
        default=None,
        description="Best guess: invoice, receipt, or other.",
    )
    total_amount: float | None = Field(default=None)
    date: str | None = Field(default=None, description="Date as printed in the document.")
    invoice_or_receipt_number: str | None = Field(default=None)
    seller_gstin: str | None = Field(
        default=None,
        description="Seller/vendor GSTIN printed with the seller's own address, usually near the top.",
    )
    buyer_gstin: str | None = Field(
        default=None,
        description="Buyer GSTIN printed in the buyer or bill-to section. Do not confuse it with seller GSTIN.",
    )
    line_items: list[ExtractedLineItem] = Field(default_factory=list)


def extract_fields(
    raw_ocr_text: str,
    *,
    image_path: str | Path | None = None,
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
) -> ExtractedDocumentFields:
    """Extract structured fields from OCR text and its source page image.

    The page image is the source of truth and OCR text is a hint. A missing API
    key is reported explicitly instead of returning a success-shaped fallback.
    """

    if not raw_ocr_text.strip() and image_path is None:
        return ExtractedDocumentFields()

    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for LLM field extraction")

    prompt = """Extract fields from the provided document page image. Treat the
image as the source of truth and the OCR text below only as a hint. The OCR may
contain spelling errors, missing punctuation, merged words, and columns in the
wrong order. Read values from the image wherever possible.

Use the seller/vendor GSTIN printed with the seller's own address (usually at
the top) for `seller_gstin`. Use the GSTIN in the buyer or bill-to section for
`buyer_gstin`. Never swap or infer one from the other.

For every line item, `net_amount` and legacy `total` mean the Net Amount
column (the final amount for that line), and `unit_price` is the per-unit
selling rate. Extract `base_rate`, `discount`, and `taxable_amount` separately
when those columns are present. Set `taxable_amount_column_present` to true
only when the page prints a separate taxable amount/value column for line
items. A per-unit "Rate (Excl. GST)" column is not a separate taxable amount
column. Set it to false when the invoice has no separate taxable amount/value
column, and null only when the image is too unclear to determine. If the invoice
has both `Cs` (cases) and `Pcs`
columns, also extract `cases`, `pieces`, and `units_per_case` (UPC); quantity
must be (cases × UPC) + pieces. When Pcs is 0 and Cs is nonzero, use the
case-based quantity. Never copy or infer a quantity from a neighboring row.
Extract the percentage shown for each line in a `GST`, `GST @`, `Tax`, or
`Tax @` column into `gst_percent` as a numeric percentage (for example, 5 for
`5%`). Treat those header labels as equivalent. Return null when a row's
percentage is blank or unreadable; do not calculate a percentage from amounts
or copy it from another row.
Read quantities carefully from the image and ignore pen marks or scribbles.
Ignore rows that are crossed out or struck through. If a value is truly
unreadable, return null; do not guess. Return null for other fields that cannot
be determined with reasonable confidence. Include only identifiable,
non-struck-through line items. Use the printed date format and classify
`document_type` as exactly `invoice`, `receipt`, or `other` when possible.

OCR text:
---
""" + raw_ocr_text + """
---
"""
    if image_path is None:
        message: str | HumanMessage = prompt
    else:
        image_file = Path(image_path)
        image_bytes = image_file.read_bytes()
        mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
        image_data = base64.b64encode(image_bytes).decode("ascii")
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}",
                    },
                },
            ]
        )
    invoke_input = [message] if isinstance(message, HumanMessage) else message

    def invoke_for_model(selected_model: str):
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=resolved_api_key,
            temperature=0,
        )
        return llm.with_structured_output(ExtractedDocumentFields).invoke(invoke_input)

    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL")
    fallback_invoke = (
        (lambda: invoke_for_model(fallback_model))
        if fallback_model and fallback_model != model
        else None
    )
    result = invoke_gemini_with_retries(
        lambda: invoke_for_model(model),
        operation="field extraction",
        fallback_invoke=fallback_invoke,
    )
    if isinstance(result, ExtractedDocumentFields):
        return result
    return ExtractedDocumentFields.model_validate(result)


def _line_quantity(line: ExtractedLineItem) -> float | None:
    if line.cases is None or line.pieces is None or line.units_per_case is None:
        return line.quantity
    return line.cases * line.units_per_case + line.pieces


def _taxable_matches(line: ExtractedLineItem) -> bool | None:
    quantity = _line_quantity(line)
    base_rate = line.base_rate if line.base_rate is not None else line.unit_price
    if quantity is None or base_rate is None or line.taxable_amount is None:
        return None

    expected_taxable = quantity * base_rate - (line.discount or 0)
    tolerance = max(abs(line.taxable_amount) * 0.02, 0.01)
    return abs(expected_taxable - line.taxable_amount) <= tolerance


def _ocr_row_percentages(
    row: list[dict],
    base_rate: float | None,
) -> set[float]:
    row_text = " ".join(
        str(word.get("text") or "")
        for word in sorted(row, key=lambda item: int(item.get("left", 0)))
    )
    percentages: set[float] = set()
    base_rate_digits = (
        re.sub(r"\D", "", f"{base_rate:.2f}")
        if base_rate is not None
        else ""
    )
    for match in re.finditer("%", row_text):
        prefix = row_text[:match.start()]
        prefix_digits = re.sub(r"\D", "", prefix)
        rate_position = (
            prefix_digits.rfind(base_rate_digits)
            if base_rate_digits
            else -1
        )
        if rate_position >= 0:
            suffix = prefix_digits[rate_position + len(base_rate_digits):]
            if 1 <= len(suffix) <= 2:
                value = float(suffix)
                if (
                    len(suffix) == 2
                    and suffix.startswith("1")
                    and value not in _STANDARD_GST_RATES
                    and float(suffix[-1]) in _STANDARD_GST_RATES
                ):
                    value = float(suffix[-1])
                if value <= 100:
                    percentages.add(value)
                    continue
        number_match = re.search(
            r"(?<![\d.])(\d{1,2}(?:\.\d+)?)\s*$",
            prefix,
        )
        if number_match is None:
            continue
        value = float(number_match.group(1))
        if 0 <= value <= 100:
            percentages.add(value)
    return percentages


def supplement_line_items_from_ocr(
    fields: ExtractedDocumentFields,
    ocr_words: list[dict],
) -> ExtractedDocumentFields:
    """Recover visible row-level tax rates missed by structured LLM output."""
    rows: list[list[dict]] = []
    for word in sorted(
        ocr_words,
        key=lambda item: (
            float(item.get("top", 0)) + float(item.get("height", 0)) / 2,
            int(item.get("left", 0)),
        ),
    ):
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        center_y = float(word.get("top", 0)) + float(word.get("height", 0)) / 2
        if not rows:
            rows.append([word])
            continue
        previous = rows[-1]
        previous_center_y = sum(
            float(item.get("top", 0)) + float(item.get("height", 0)) / 2
            for item in previous
        ) / len(previous)
        if abs(center_y - previous_center_y) <= 28:
            previous.append(word)
        else:
            rows.append([word])

    used_rows: set[int] = set()
    previous_row_y = -1.0
    enriched_lines: list[ExtractedLineItem] = []
    ignored_terms = {
        "bag", "gm", "kg", "loose", "normal", "packet", "pouch", "tin",
    }
    for line in fields.line_items:
        terms = {
            token
            for token in re.findall(r"[a-z]+", (line.description or "").casefold())
            if len(token) > 2 and token not in ignored_terms
        }
        best_row_index: int | None = None
        best_score = 0
        best_row_y = previous_row_y
        for row_index, row in enumerate(rows):
            if row_index in used_rows:
                continue
            row_y = min(
                float(word.get("top", 0)) + float(word.get("height", 0)) / 2
                for word in row
            )
            if row_y <= previous_row_y:
                continue
            row_terms = {
                token
                for word in row
                for token in re.findall(
                    r"[a-z]+",
                    str(word.get("text") or "").casefold(),
                )
            }
            score = len(terms & row_terms)
            if score > best_score:
                best_row_index = row_index
                best_score = score
                best_row_y = row_y

        gst_percent = line.gst_percent
        if best_row_index is not None:
            used_rows.add(best_row_index)
            previous_row_y = best_row_y
            percentages = _ocr_row_percentages(
                rows[best_row_index],
                line.base_rate,
            )
            if gst_percent is None and len(percentages) == 1:
                gst_percent = percentages.pop()

        quantity = _line_quantity(line)
        taxable_amount = line.taxable_amount
        if (
            taxable_amount is None
            and gst_percent is not None
            and fields.taxable_amount_column_present is not True
            and quantity is not None
            and line.base_rate is not None
        ):
            taxable_amount = round(
                quantity * line.base_rate - (line.discount or 0),
                2,
            )

        enriched_lines.append(
            line.model_copy(
                update={
                    "gst_percent": gst_percent,
                    "taxable_amount": taxable_amount,
                }
            )
        )

    return fields.model_copy(update={"line_items": enriched_lines})


def _reextract_line_item(
    raw_ocr_text: str,
    line: ExtractedLineItem,
    image_path: str | Path,
    *,
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
) -> ExtractedLineItem:
    resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not resolved_api_key:
        raise RuntimeError("GEMINI_API_KEY is required for line-item re-extraction")

    image_file = Path(image_path)
    image_data = base64.b64encode(image_file.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(image_file.name)[0] or "image/png"
    prompt = f"""Re-extract only this invoice line item from the page image. Use
the image as the source of truth. Do not borrow any value, especially quantity,
from a neighboring row. For Cs and Pcs columns, calculate quantity as
(cases × units_per_case) + pieces when all three values are visible. Return
null for unreadable values. Preserve the line description if it matches this
row, and return the row's base_rate, unit_price, discount, taxable_amount,
gst_percent (from GST/GST @/Tax/Tax @ columns), net_amount, and legacy total
where visible. Return null if the row's tax percentage is blank or unreadable;
do not infer or copy a neighboring row's percentage.

OCR hint for the page:
---
{raw_ocr_text}
---

Previously extracted target row:
{line.model_dump_json()}
"""
    invoke_input = [
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_data}",
                    },
                },
            ]
        )
    ]

    def invoke_for_model(selected_model: str):
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=resolved_api_key,
            temperature=0,
        )
        return llm.with_structured_output(ExtractedLineItem).invoke(invoke_input)

    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL")
    fallback_invoke = (
        (lambda: invoke_for_model(fallback_model))
        if fallback_model and fallback_model != model
        else None
    )
    result = invoke_gemini_with_retries(
        lambda: invoke_for_model(model),
        operation="line-item re-extraction",
        fallback_invoke=fallback_invoke,
    )
    if isinstance(result, ExtractedLineItem):
        return result
    return ExtractedLineItem.model_validate(result)


def validate_line_items(
    fields: ExtractedDocumentFields,
    raw_ocr_text: str,
    image_path: str | Path,
    *,
    api_key: str | None = None,
    model: str = "gemini-3.5-flash-lite",
) -> ExtractedDocumentFields:
    """Derive case quantities and retry arithmetic mismatches once per line."""
    checked_lines: list[ExtractedLineItem] = []
    for original in fields.line_items:
        line = original.model_copy(
            update={
                "quantity": _line_quantity(original),
                "net_amount": (
                    original.net_amount
                    if original.net_amount is not None
                    else original.total
                ),
            }
        )
        match = _taxable_matches(line)
        if match is False:
            line = _reextract_line_item(
                raw_ocr_text,
                line,
                image_path,
                api_key=api_key,
                model=model,
            )
            line = line.model_copy(
                update={
                    "quantity": _line_quantity(line),
                    "net_amount": (
                        line.net_amount if line.net_amount is not None else line.total
                    ),
                    "needs_review": _taxable_matches(line) is not True,
                }
            )
        checked_lines.append(line)
    return fields.model_copy(update={"line_items": checked_lines})
