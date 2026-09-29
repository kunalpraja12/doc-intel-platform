# Doc Intel Platform

Doc Intel Platform is a production-style AI/ML portfolio project for processing real-world documents such as invoices, receipts, insurance claims, ID cards, and forms. The platform is designed to demonstrate end-to-end engineering skills for document understanding, data validation, orchestration, and retrieval-augmented chat over business documents.

## Goals

- Ingest unstructured and semi-structured documents from multiple sources
- Extract text and structure using OCR + CV preprocessing + layout detection
- Classify and enrich document data with ML/LLM workflows
- Store trusted metadata and extracted records in PostgreSQL
- Support natural-language Q&A over a document corpus using RAG
- Process uploaded documents synchronously through OCR and extraction
- Provide a demo-friendly API and UI for upload, review, and chat

## Architecture Summary

This repository is intentionally scaffolded as a clean Python service-oriented project so each subsystem can be built incrementally:

- `app/` contains the FastAPI application, route layer, and service entrypoints
- `db/` contains database models, session management, and repository patterns
- `pipeline/` contains OCR/CV, validation, orchestration, and RAG logic
- `models/` contains ML model wrappers and prompt assets
- `tests/` contains unit and integration tests
- `docker/` and root config files provide local orchestration for dev and demo usage

## Proposed Folder Structure

```text
.
├── README.md
├── .gitignore
├── .env.example
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   └── health.py
│   │   └── schemas/
│   │       ├── __init__.py
│   │       ├── document.py
│   │       └── chat.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── security.py
│   │   └── constants.py
├── db/
│   ├── __init__.py
│   ├── session.py
│   ├── alembic.ini
│   ├── models/
│   │   ├── __init__.py
│   │   ├── document.py
│   │   ├── extraction.py
│   │   ├── user.py
│   │   └── audit.py
│   ├── repositories/
│   │   ├── __init__.py
│   │   └── document_repo.py
│   └── migrations/
│       └── README.md
├── pipeline/
│   ├── __init__.py
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── ocr.py
│   │   ├── cv_preprocessing.py
│   │   └── layout.py
│   ├── validate/
│   │   ├── __init__.py
│   │   ├── rules.py
│   │   └── llm_enrichment.py
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   └── states.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── utils/
│       ├── __init__.py
│       └── io.py
├── models/
│   ├── __init__.py
│   ├── document_classifier.py
│   ├── ner.py
│   └── prompts/
│       ├── __init__.py
│       ├── extraction_prompt.txt
│       └── qa_prompt.txt
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── __init__.py
│   │   └── test_extraction.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_pipeline.py
│   └── fixtures/
│       ├── sample_invoice.pdf
│       └── sample_receipt.jpg
├── scripts/
│   ├── bootstrap.py
│   └── seed_data.py
├── docs/
│   └── architecture.md
└── .github/
    └── workflows/
        └── ci.yml
```

## Why this structure?

This layout follows Python backend best practices for a production AI service:

- `app/` separates HTTP and application concerns from domain logic
- `core/` keeps configuration, security, and shared settings central
- `workers/` isolates asynchronous job processing from request handling
- `db/` cleanly separates persistence logic and model definitions
- `pipeline/` groups document intelligence stages in a modular, testable way
- `models/` keeps inference code and prompt templates separate from orchestration
- `tests/` supports both fast unit tests and end-to-end integration validation

## Next steps

This scaffold is intentionally sparse and placeholder-based. The next milestones are:

1. Define the relational schema and migration strategy
2. Implement the OCR + CV preprocessing pipeline
3. Build the LangGraph document processing orchestrator
4. Add the RAG chat layer and retrieval pipeline
5. Expose the API and demo UI
6. Add Docker, monitoring, and robust tests

This repository is ready for incremental implementation, feature by feature.
