"""Chat endpoints for document-grounded Q&A.

TODO: Add chat history, retrieval, and natural-language question handling.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def ask_question() -> dict[str, str]:
    return {"answer": "Chat layer is not implemented yet."}
