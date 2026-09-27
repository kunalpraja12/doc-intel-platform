"""Add pgvector storage for page-level document embeddings."""

from alembic import op
import sqlalchemy as sa

revision = "0003_add_document_embeddings"
down_revision = "0002_add_metadata_json"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "document_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "document_id",
            sa.Integer(),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("embedding", sa.Text(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    # SQLAlchemy has no built-in pgvector type; alter the placeholder text
    # column to the required 768-dimensional vector type.
    op.execute(
        "ALTER TABLE document_embeddings "
        "ALTER COLUMN embedding TYPE vector(768) "
        "USING embedding::vector"
    )
    op.create_index(
        "ix_document_embeddings_document_id",
        "document_embeddings",
        ["document_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_document_embeddings_document_id", table_name="document_embeddings")
    op.drop_table("document_embeddings")
