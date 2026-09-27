"""Add SHA-256 content hashes to documents for duplicate detection."""

from alembic import op
import sqlalchemy as sa

revision = "0004_add_document_content_hash"
down_revision = "0003_add_document_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("content_hash", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_documents_content_hash",
        "documents",
        ["content_hash"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("ix_documents_content_hash", table_name="documents")
    op.drop_column("documents", "content_hash")
