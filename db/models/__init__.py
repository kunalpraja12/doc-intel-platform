"""Relational models package.

Importing this module registers all model classes with SQLAlchemy's declarative
base so relationships that reference classes by name can be resolved.
"""

# Import all models so that the declarative base sees every mapped class.
# This module is intentionally side-effecting: import db.models early in
# application startup (or before importing individual model modules) to
# ensure mappers are configured.
from db.models.document import Document  # noqa: F401
from db.models.extraction import ExtractionResult  # noqa: F401
from db.models.line_item import LineItem  # noqa: F401
from db.models.anomaly_flag import AnomalyFlag  # noqa: F401
from db.models.user import User  # noqa: F401
from db.models.audit import AuditLog  # noqa: F401

__all__ = [
    "Document",
    "ExtractionResult",
    "LineItem",
    "AnomalyFlag",
    "User",
    "AuditLog",
]
