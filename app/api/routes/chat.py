"""Single-turn RAG chat over uploaded document page embeddings."""

from __future__ import annotations

import os
import logging
import math
import re
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from langchain_google_genai import ChatGoogleGenerativeAI

from app.api.schemas.chat import ChatQueryResponse, ChatSource
from app.core.errors import (
    invoke_gemini_with_retries,
    is_gemini_quota_error,
    is_gemini_temporary_error,
    user_friendly_message,
)
from app.dependencies import get_db
from db.models.document import Document
from db.repositories.document_repo import DocumentRepository
from pipeline.extract.embeddings import embed_text

load_dotenv()

router = APIRouter(prefix="/chat", tags=["chat"])
repo = DocumentRepository()
logger = logging.getLogger(__name__)

MAX_ESTIMATED_PROMPT_TOKENS = 60_000
MAX_SUMMARY_DOCUMENTS = 10
MAX_LISTED_LINE_ITEMS = 100
SYSTEM_PROMPT = """You are a helpful, conversational assistant answering a user's
question about their documents. Answer directly in short, clear, natural
sentences. Do not begin with phrases like "Based on the provided context" or
"According to the documents". If the answer is a single amount or fact, state
it plainly. Avoid bullet points or a formal report unless the question genuinely
calls for a list.

Use only the document context below. If it does not contain enough information,
say so naturally and do not invent details. Sources are returned separately by
the API; do not recite source metadata unless it helps answer the question.
Never say a document is "the most recent" or "the latest uploaded" unless
the supplied context explicitly includes that fact. When the user has not
identified a particular document, name the documents you used in your answer.
When context explicitly says a list covers only some documents, state that
coverage limit and explain that the user can narrow by vendor, invoice number,
or "Just the last upload" to get a more focused answer."""


def _estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def _available(value: Any) -> Any:
    return value if value is not None else "not available"


def _available_item_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: _available(value)
        for key, value in item.items()
        if key != "document_id"
    }


def _compose_prompt(question: str, contexts: list[str]) -> str:
    context_text = "\n\n".join(contexts)
    prefix = f"{SYSTEM_PROMPT}\n\nDocument context:\n---\n"
    suffix = f"\n---\n\nQuestion: {question}\n"
    truncation_notice = "\n[Context truncated to stay within the token budget.]"
    max_context_chars = max(
        0,
        MAX_ESTIMATED_PROMPT_TOKENS * 4 - len(prefix) - len(suffix),
    )
    if len(context_text) > max_context_chars:
        max_context_chars = max(0, max_context_chars - len(truncation_notice))
        context_text = (
            context_text[:max_context_chars]
            + truncation_notice
        )
    prompt = f"{prefix}{context_text}{suffix}"
    estimated_tokens = _estimate_tokens(prompt)
    if estimated_tokens > MAX_ESTIMATED_PROMPT_TOKENS:
        raise RuntimeError("Unable to fit chat prompt within the configured token budget")
    logger.info("Chat request estimated input tokens: %s", estimated_tokens)
    return prompt


def _answer_with_context(question: str, contexts: list[str]) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required for chat")
    prompt = _compose_prompt(question, contexts)
    model = "gemini-3.5-flash-lite"

    def invoke_for_model(selected_model: str):
        llm = ChatGoogleGenerativeAI(
            model=selected_model,
            google_api_key=api_key,
            temperature=0,
        )
        return llm.invoke(prompt)

    fallback_model = os.getenv("GEMINI_FALLBACK_MODEL")
    fallback_invoke = (
        (lambda: invoke_for_model(fallback_model))
        if fallback_model and fallback_model != model
        else None
    )
    response = invoke_gemini_with_retries(
        lambda: invoke_for_model(model),
        operation="chat answer",
        fallback_invoke=fallback_invoke,
    )
    return _plain_text_content(response.content)


def _plain_text_content(content: Any) -> str:
    """Normalize LangChain text content blocks to the answer string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block
            if isinstance(block, str)
            else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") == "text")
        ]
        answer = "".join(text_parts).strip()
        if answer:
            return answer
    raise RuntimeError("Gemini returned no plain text answer")


def _needs_structured_data(question: str) -> bool:
    normalized = question.casefold()
    has_structured_value_term = any(
        term in normalized
        for term in ("item", "items", "line item", "line items", "total", "totals", "spent", "sum")
    )
    asks_for_count = "how many" in normalized or "count" in normalized
    asks_to_count_documents = asks_for_count and any(
        term in normalized for term in ("invoice", "invoices", "receipt", "receipts", "documents")
    )
    return has_structured_value_term or asks_to_count_documents


def _is_list_items_question(question: str) -> bool:
    normalized = question.casefold()
    asks_for_items = any(term in normalized for term in ("item", "items", "line item", "line items"))
    asks_for_list = any(term in normalized for term in ("list", "show", "all", "every"))
    return asks_for_items and asks_for_list and not (
        "how many" in normalized or "count" in normalized
    )


def _document_ids_mentioned_in_question(
    question: str,
    documents: list[dict[str, Any]],
) -> list[int] | None:
    ignored_terms = {
        "all", "amount", "are", "bought", "documents", "from", "how", "invoice",
        "invoices", "item", "items", "list", "many", "much", "receipt", "receipts",
        "show", "spent", "spending", "the", "total", "totals", "what", "with",
    }
    question_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", question.casefold())
        if len(term) > 2 and term not in ignored_terms
    }
    if not question_terms:
        return None

    matches = []
    for document in documents:
        identifying_text = " ".join(
            str(document.get(key) or "")
            for key in ("vendor_name", "invoice_number", "file_name")
        )
        identifying_terms = {
            term
            for term in re.findall(r"[a-z0-9]+", identifying_text.casefold())
            if len(term) > 2 and term not in ignored_terms
        }
        if question_terms & identifying_terms:
            matches.append(int(document["document_id"]))
    if matches and len(matches) < len(documents):
        return matches
    return None





@router.post("/query", response_model=ChatQueryResponse)
def query_chat(
    question: str = Body(..., media_type="text/plain"),
    document_ids: list[int] | None = Query(default=None),
    db: Session = Depends(get_db),
) -> ChatQueryResponse:
    try:
        logger.info(
            "Chat request question token estimate: %s",
            _estimate_tokens(question),
        )
        contexts: list[str] = []
        sources: list[ChatSource] = []
        if _needs_structured_data(question):
            overview = repo.get_structured_chat_overview(db, document_ids)
            documents = overview["documents"]
            mentioned_ids = _document_ids_mentioned_in_question(question, documents)
            if mentioned_ids is not None:
                overview = repo.get_structured_chat_overview(db, mentioned_ids)
                documents = overview["documents"]
            if not documents:
                return ChatQueryResponse(
                    answer="I couldn't find matching documents with structured data.",
                    sources=[],
                )

            list_items = _is_list_items_question(question)
            if list_items:
                selected_documents = documents[:MAX_SUMMARY_DOCUMENTS]
                selected_ids = [
                    int(document["document_id"])
                    for document in selected_documents
                ]
                line_items = repo.get_line_items_for_documents(
                    db,
                    selected_ids,
                    limit=MAX_LISTED_LINE_ITEMS,
                )
                item_groups: dict[int, list[dict[str, Any]]] = {}
                for item in line_items:
                    item_groups.setdefault(item["document_id"], []).append(item)
                contexts.append(
                    f"Compact item results cover {len(selected_documents)} of "
                    f"{overview['document_count']} matching documents. The database "
                    f"query includes at most {MAX_LISTED_LINE_ITEMS} line items. "
                    "For each included document, show its vendor, invoice number, "
                    "total, and the available item descriptions, quantities, and prices. "
                    "Be explicit that this is a partial list if fewer documents or "
                    "items are shown than exist."
                )
                for document in selected_documents:
                    contexts.append(
                        f"Document {document['document_id']} ({document['file_name']}): "
                        f"vendor={_available(document['vendor_name'])}, "
                        f"invoice number={_available(document['invoice_number'])}, "
                        f"total={_available(document['total_amount'])}; "
                        f"items={[_available_item_fields(item) for item in item_groups.get(document['document_id'], [])]}"
                    )
                if (
                    overview["document_count"] > len(selected_documents)
                    or overview["line_item_count"] > len(line_items)
                ):
                    contexts.append(
                        "This is a partial result. Suggest narrowing by vendor, "
                        "invoice number, or selecting 'Just the last upload'."
                    )
                context_documents = selected_documents
            else:
                names = ", ".join(
                    document["file_name"]
                    for document in documents[:MAX_SUMMARY_DOCUMENTS]
                )
                contexts.append(
                    "Computed SQL aggregates (not raw line-item rows): "
                    f"documents={overview['document_count']}, "
                    f"invoices={overview['invoice_count']}, "
                    f"receipts={overview['receipt_count']}, "
                    f"line_items={overview['line_item_count']}, "
                    f"sum_of_quantities={_available(overview['quantity_total'])}, "
                    f"sum_of_line_net_amounts={_available(overview['line_net_total'])}, "
                    f"sum_of_document_totals={overview['document_total']}. "
                    f"Documents included in the aggregates: {overview['document_count']}. "
                    f"Document names for attribution: {names or 'none'}."
                )
                context_documents = documents[:MAX_SUMMARY_DOCUMENTS]
            for document in context_documents:
                sources.append(
                    ChatSource(
                        document_id=str(document["document_id"]),
                        file_name=document["file_name"],
                    )
                )
        else:
            query_embedding = embed_text(question)
            matches = repo.search_embeddings(
                db,
                query_embedding,
                limit=5,
                document_ids=document_ids,
            )
            if not matches:
                return ChatQueryResponse(
                    answer="No matching document embeddings are available for retrieval yet.",
                    sources=[],
                )
            for embedding, distance in matches:
                document = db.get(Document, embedding.document_id)
                if not document:
                    continue
                page_label = (
                    f"page {embedding.page_number}"
                    if embedding.page_number is not None
                    else "document"
                )
                contexts.append(
                    f"Document {document.id} ({document.file_name}), {page_label}, "
                    f"similarity distance {distance:.4f}:\n{embedding.chunk_text}"
                )
                sources.append(
                    ChatSource(
                        document_id=str(document.id),
                        file_name=document.file_name,
                        page_number=embedding.page_number,
                    )
                )
        answer = _answer_with_context(question, contexts)
        return ChatQueryResponse(answer=answer, sources=sources)
    except Exception as exc:
        logger.exception("Chat query failed")
        if isinstance(exc, SQLAlchemyError):
            raise HTTPException(
                status_code=503,
                detail=user_friendly_message(exc),
            ) from exc
        if is_gemini_quota_error(exc):
            status_code = 429
        elif is_gemini_temporary_error(exc):
            status_code = 503
        else:
            status_code = 500
        raise HTTPException(
            status_code=status_code,
            detail=user_friendly_message(exc, ai_operation=True),
        ) from exc
