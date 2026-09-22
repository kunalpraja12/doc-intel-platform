"""Async tasks for OCR, enrichment, and document processing jobs.

TODO: Define Celery tasks for OCR, validation, and RAG indexing.
"""

from celery import shared_task


@shared_task
def process_document(document_id: str) -> str:
    return f"Processing document {document_id}"
