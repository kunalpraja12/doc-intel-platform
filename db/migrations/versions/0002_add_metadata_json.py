"""Add metadata_json column to documents

Adds the metadata_json JSON column to documents to match the current model.
If an older 'metadata' column exists, its values are copied into the new
metadata_json column for compatibility.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0002_add_metadata_json'
down_revision = '0001_initial_schema'
branch_labels = None
depend_on = None


def upgrade():
    # Use a safe ALTER TABLE ... ADD COLUMN IF NOT EXISTS to avoid errors
    op.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata_json JSON;")
    # If an older 'metadata' column exists, copy its contents into metadata_json
    op.execute(
        "DO $$\nBEGIN\nIF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='documents' AND column_name='metadata') THEN\n    UPDATE documents SET metadata_json = metadata WHERE metadata IS NOT NULL;\nEND IF;\nEND$$;"
    )


def downgrade():
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS metadata_json;")
