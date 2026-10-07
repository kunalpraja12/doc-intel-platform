import asyncio
from io import BytesIO
from threading import Event

import pytest
from starlette.datastructures import Headers, UploadFile

from app.api.routes import documents
from app.api.schemas.document import UploadFileResult


@pytest.mark.asyncio
async def test_upload_processing_does_not_block_event_loop(monkeypatch):
    event_loop_resumed = Event()

    def process_upload(contents, filename, content_type):
        if not event_loop_resumed.wait(timeout=2):
            raise AssertionError("Upload blocked the event loop")
        return UploadFileResult(file_name=filename, status="completed")

    async def resume_event_loop():
        await asyncio.sleep(0)
        event_loop_resumed.set()

    monkeypatch.setattr(documents, "_process_document_upload", process_upload)
    upload = UploadFile(
        filename="invoice.jpg",
        file=BytesIO(b"test image"),
        headers=Headers({"content-type": "image/jpeg"}),
    )

    results, _ = await asyncio.gather(
        documents.upload_document([upload]),
        resume_event_loop(),
    )

    assert results[0].status == "completed"
