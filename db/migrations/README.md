# Database migrations

This directory is reserved for Alembic migration scripts.

Planned migrations include:
- initial schema for documents, users, extraction_results, line_items, and anomaly_flags
- audit log table for validation and review events
- (deferred) search index and vector metadata tables for RAG

When ready, generate migration scripts with `alembic revision --autogenerate -m "initial schema"` and verify the output before applying.
