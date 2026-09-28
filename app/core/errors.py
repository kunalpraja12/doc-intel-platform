"""Safe user-facing error messages and retry handling for Gemini requests."""

from __future__ import annotations

import logging
import math
import re
import time
from collections.abc import Callable
from typing import TypeVar

from sqlalchemy.exc import SQLAlchemyError

T = TypeVar("T")

AI_BUSY_MESSAGE = "The AI service is busy right now. Please try again in a minute."
AI_QUOTA_MESSAGE = (
    "The AI service is at its free-tier limit. Please wait about a minute and try again."
)
DATABASE_MESSAGE = "We couldn't reach the database. Please try again in a moment."
FILE_MESSAGE = "This file couldn't be read. Please check it and try another file."
FILE_TOO_LARGE_MESSAGE = "This file is larger than 20MB."
GENERIC_MESSAGE = "Something went wrong. Please try again."

logger = logging.getLogger(__name__)


def _status_code(error: BaseException) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(error, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            normalized = str(value).casefold()
            status_names = {
                "resource_exhausted": 429,
                "unavailable": 503,
                "internal": 500,
                "unknown": 500,
                "deadline_exceeded": 504,
            }
            for name, status in status_names.items():
                if name in normalized:
                    return status
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    try:
        if response_status is not None:
            return int(response_status)
    except (TypeError, ValueError):
        pass
    status_match = re.search(r"\b(429|500|502|503|504)\b", str(error))
    return int(status_match.group(1)) if status_match else None


def retry_delay_seconds(error: BaseException) -> int | None:
    message = str(error)
    patterns = (
        r"retryDelay['\"\s:=]+([0-9]+(?:\.[0-9]+)?)s",
        r"retry[_ ]delay['\"\s:=]+([0-9]+(?:\.[0-9]+)?)s",
        r"retry[_ ]after['\"\s:=]+([0-9]+(?:\.[0-9]+)?)s",
        r"retryDelay['\"]?\s*:\s*\{[^}]*['\"]seconds['\"]\s*:\s*([0-9]+)",
        r"['\"]seconds['\"]\s*:\s*([0-9]+)",
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*seconds?",
        r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s\b",
        r"seconds['\"\s:=]+([0-9]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, message, flags=re.IGNORECASE)
        if match:
            return max(1, math.ceil(float(match.group(1))))
    return None


def _is_database_error(error: BaseException) -> bool:
    if isinstance(error, SQLAlchemyError):
        return True
    module = type(error).__module__.casefold()
    name = type(error).__name__.casefold()
    message = str(error).casefold()
    if any(package in module for package in ("psycopg", "asyncpg", "sqlalchemy")):
        return True
    if any(term in name for term in ("operationalerror", "interfaceerror", "databaseerror")):
        return True
    return any(
        term in message
        for term in (
            "connection has been closed",
            "could not connect to server",
            "server closed the connection",
            "ssl connection has been closed",
            "consuming input failed",
            "idle in transaction",
        )
    )


def is_gemini_quota_error(error: BaseException) -> bool:
    message = str(error).casefold()
    return _status_code(error) == 429 or "resource_exhausted" in message


def _is_gemini_error(error: BaseException) -> bool:
    module = type(error).__module__.casefold()
    message = str(error).casefold()
    return (
        "google" in module
        or "gemini" in module
        or "langchain_google_genai" in module
        or "resource_exhausted" in message
        or "gemini" in message
    )


def is_gemini_temporary_error(error: BaseException) -> bool:
    status = _status_code(error)
    if status in {500, 502, 503, 504}:
        return True
    if (
        isinstance(error, TimeoutError)
        or "timeout" in type(error).__name__.casefold()
        or "timed out" in str(error).casefold()
        or "timeout" in str(error).casefold()
    ):
        return True
    return "deadline exceeded" in str(error).casefold()


def user_friendly_message(
    error: BaseException,
    *,
    ai_operation: bool = False,
    upload: bool = False,
) -> str:
    """Map internal exceptions to a short, safe message without exposing details."""
    if _is_database_error(error):
        return DATABASE_MESSAGE

    if (ai_operation or _is_gemini_error(error)) and is_gemini_quota_error(error):
        delay = retry_delay_seconds(error)
        if delay is not None:
            suffix = "second" if delay == 1 else "seconds"
            return (
                "The AI service is at its free-tier limit. "
                f"Please wait {delay} {suffix} and try again."
            )
        return AI_QUOTA_MESSAGE

    if (ai_operation or _is_gemini_error(error)) and is_gemini_temporary_error(error):
        return AI_BUSY_MESSAGE

    if upload:
        message = str(error).casefold()
        if "20mb" in message or "file exceeds" in message or "too large" in message:
            return FILE_TOO_LARGE_MESSAGE
        return FILE_MESSAGE

    return GENERIC_MESSAGE


def is_friendly_message(message: object) -> bool:
    if not isinstance(message, str):
        return False
    if message in {
        AI_BUSY_MESSAGE,
        AI_QUOTA_MESSAGE,
        DATABASE_MESSAGE,
        FILE_MESSAGE,
        FILE_TOO_LARGE_MESSAGE,
        GENERIC_MESSAGE,
    }:
        return True
    return bool(
        re.fullmatch(
            r"The AI service is at its free-tier limit\. Please wait \d+ seconds? and try again\.",
            message,
        )
    )


def invoke_gemini_with_retries(
    invoke: Callable[[], T],
    *,
    operation: str,
    fallback_invoke: Callable[[], T] | None = None,
) -> T:
    """Retry transient model failures, then optionally use one fallback model."""
    quota_retried = False
    temporary_delays = (2, 5)
    temporary_retry_index = 0

    while True:
        try:
            return invoke()
        except Exception as error:
            if is_gemini_quota_error(error):
                if not quota_retried:
                    quota_retried = True
                    logger.warning("Gemini quota retry for %s", operation, exc_info=True)
                    continue
                raise

            if is_gemini_temporary_error(error):
                if temporary_retry_index < len(temporary_delays):
                    delay = temporary_delays[temporary_retry_index]
                    temporary_retry_index += 1
                    logger.warning(
                        "Retrying Gemini %s after temporary failure in %s seconds",
                        operation,
                        delay,
                        exc_info=True,
                    )
                    time.sleep(delay)
                    continue

                if (
                    _status_code(error) == 503
                    and fallback_invoke is not None
                ):
                    logger.warning(
                        "Primary Gemini model remained unavailable for %s; trying fallback model",
                        operation,
                        exc_info=True,
                    )
                    return fallback_invoke()
            raise
