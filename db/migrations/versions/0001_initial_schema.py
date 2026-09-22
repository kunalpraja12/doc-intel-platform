"""Initial schema migration

Creates tables: users, documents, extraction_results, line_items, anomaly_flags
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_initial_schema'
down_revision = None
branch_labels = None
depend_on = None


def upgrade():
    # users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('email', sa.String(length=255), nullable=False, unique=True),
        sa.Column('full_name', sa.String(length=255), nullable=False),
    )

    # documents table
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('file_name', sa.String(length=1024), nullable=False),
        sa.Column('file_path', sa.String(length=2048), nullable=True),
        sa.Column('content_type', sa.String(length=255), nullable=True),
        sa.Column('file_size', sa.Integer, nullable=True),
        sa.Column('document_type', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='uploaded'),
        sa.Column('uploaded_by', sa.String(length=255), nullable=True),
        sa.Column('metadata_json', sa.JSON, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
    )

    # extraction_results table
    op.create_table(
        'extraction_results',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('document_id', sa.Integer, sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('extracted_data', sa.JSON, nullable=True),
        sa.Column('confidence_score', sa.Float, nullable=True),
        sa.Column('model_name', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )

    # line_items table
    op.create_table(
        'line_items',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('document_id', sa.Integer, sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('quantity', sa.Float, nullable=True),
        sa.Column('unit_price', sa.Float, nullable=True),
        sa.Column('total', sa.Float, nullable=True),
        sa.Column('raw', sa.JSON, nullable=True),
    )

    # anomaly_flags table
    op.create_table(
        'anomaly_flags',
        sa.Column('id', sa.Integer, primary_key=True, index=True),
        sa.Column('document_id', sa.Integer, sa.ForeignKey('documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('field', sa.String(length=255), nullable=True),
        sa.Column('severity', sa.String(length=50), nullable=True),
        sa.Column('message', sa.Text, nullable=True),
        sa.Column('resolved', sa.Boolean, nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()')),
    )


def downgrade():
    op.drop_table('anomaly_flags')
    op.drop_table('line_items')
    op.drop_table('extraction_results')
    op.drop_table('documents')
    op.drop_table('users')
