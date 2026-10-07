"""Add per-line GST percentage."""

from alembic import op
import sqlalchemy as sa

revision = "0005_add_line_item_gst_percent"
down_revision = "0004_add_document_content_hash"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "line_items",
        sa.Column("gst_percent", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("line_items", "gst_percent")
