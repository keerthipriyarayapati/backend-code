"""001_initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-09-11 22:45:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. loan_applications
    op.create_table(
        'loan_applications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('loan_type', sa.String(length=100), nullable=False),
        sa.Column('applicant_name', sa.String(length=200), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='NOT_STARTED'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('application_id')
    )
    op.create_index(op.f('ix_loan_applications_application_id'), 'loan_applications', ['application_id'], unique=True)

    # 2. applicants
    op.create_table(
        'applicants',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('applicant_type', sa.String(length=50), nullable=False, server_default='PRIMARY_APPLICANT'),
        sa.Column('full_name', sa.String(length=200), nullable=True),
        sa.Column('date_of_birth', sa.String(length=50), nullable=True),
        sa.Column('gender', sa.String(length=20), nullable=True),
        sa.Column('email', sa.String(length=150), nullable=True),
        sa.Column('phone', sa.String(length=50), nullable=True),
        sa.Column('address', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. document_requirements
    op.create_table(
        'document_requirements',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('loan_type', sa.String(length=100), nullable=False),
        sa.Column('document_type', sa.String(length=100), nullable=False),
        sa.Column('display_name', sa.String(length=200), nullable=False),
        sa.Column('purpose', sa.Text(), nullable=True),
        sa.Column('requirement_status', sa.String(length=50), nullable=False, server_default='REQUIRED'),
        sa.Column('applicant_role', sa.String(length=50), nullable=False, server_default='PRIMARY'),
        sa.Column('allowed_extensions', sa.String(length=200), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_document_requirements_loan_type'), 'document_requirements', ['loan_type'], unique=False)

    # 4. documents
    op.create_table(
        'documents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('applicant_id', sa.Integer(), nullable=True),
        sa.Column('requirement_id', sa.String(length=100), nullable=True),
        sa.Column('file_name', sa.String(length=255), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_extension', sa.String(length=20), nullable=True),
        sa.Column('mime_type', sa.String(length=100), nullable=True),
        sa.Column('document_type', sa.String(length=100), nullable=True),
        sa.Column('upload_status', sa.String(length=50), nullable=False, server_default='accepted'),
        sa.Column('extraction_method', sa.String(length=100), nullable=True),
        sa.Column('ocr_used', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('text_quality', sa.String(length=50), nullable=True),
        sa.Column('uploaded_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['applicant_id'], ['applicants.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 5. document_classifications
    op.create_table(
        'document_classifications',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('predicted_document_type', sa.String(length=100), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('classification_status', sa.String(length=50), nullable=False, server_default='CLASSIFIED'),
        sa.Column('loan_type', sa.String(length=100), nullable=True),
        sa.Column('classification_reason', sa.Text(), nullable=True),
        sa.Column('model_name', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 6. extracted_fields
    op.create_table(
        'extracted_fields',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=False),
        sa.Column('field_name', sa.String(length=100), nullable=False),
        sa.Column('field_value', sa.Text(), nullable=True),
        sa.Column('normalized_value', sa.Text(), nullable=True),
        sa.Column('data_type', sa.String(length=50), nullable=False, server_default='string'),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('source_page', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('source_text', sa.Text(), nullable=True),
        sa.Column('extraction_method', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_extracted_fields_field_name'), 'extracted_fields', ['field_name'], unique=False)

    # 7. validation_results
    op.create_table(
        'validation_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('document_id', sa.Integer(), nullable=True),
        sa.Column('field_name', sa.String(length=100), nullable=False),
        sa.Column('validation_type', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PASS'),
        sa.Column('severity', sa.String(length=50), nullable=False, server_default='INFO'),
        sa.Column('expected_value', sa.Text(), nullable=True),
        sa.Column('actual_value', sa.Text(), nullable=True),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # 8. cross_document_findings
    op.create_table(
        'cross_document_findings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('document_a_id', sa.Integer(), nullable=True),
        sa.Column('document_b_id', sa.Integer(), nullable=True),
        sa.Column('field', sa.String(length=100), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('severity', sa.String(length=50), nullable=False, server_default='INFO'),
        sa.Column('value_a', sa.Text(), nullable=True),
        sa.Column('value_b', sa.Text(), nullable=True),
        sa.Column('normalized_value_a', sa.Text(), nullable=True),
        sa.Column('normalized_value_b', sa.Text(), nullable=True),
        sa.Column('comparison_result', sa.String(length=50), nullable=False),
        sa.Column('explanation', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False, server_default='1.0'),
        sa.Column('evidence', sa.Text(), nullable=True),
        sa.Column('recommendation', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['document_a_id'], ['documents.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['document_b_id'], ['documents.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )

    # 9. risk_assessments
    op.create_table(
        'risk_assessments',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('risk_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('risk_level', sa.String(length=50), nullable=False, server_default='LOW'),
        sa.Column('recommended_action', sa.String(length=100), nullable=False, server_default='PROCEED_TO_REPORT'),
        sa.Column('overall_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 10. risk_factors
    op.create_table(
        'risk_factors',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('risk_assessment_id', sa.Integer(), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('severity', sa.String(length=50), nullable=False, server_default='LOW'),
        sa.Column('points', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('source', sa.String(length=100), nullable=True),
        sa.Column('document_id', sa.Integer(), nullable=True),
        sa.Column('finding_id', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['risk_assessment_id'], ['risk_assessments.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 11. final_reports
    op.create_table(
        'final_reports',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('decision', sa.String(length=50), nullable=False),
        sa.Column('decision_reason', sa.Text(), nullable=True),
        sa.Column('risk_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('risk_level', sa.String(length=50), nullable=False, server_default='LOW'),
        sa.Column('verification_coverage', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('consistency_score', sa.Float(), nullable=True),
        sa.Column('document_summary', sa.Text(), nullable=True),
        sa.Column('validation_summary', sa.Text(), nullable=True),
        sa.Column('cross_document_summary', sa.Text(), nullable=True),
        sa.Column('risk_summary', sa.Text(), nullable=True),
        sa.Column('key_findings', sa.Text(), nullable=True),
        sa.Column('recommendations', sa.Text(), nullable=True),
        sa.Column('executive_summary', sa.Text(), nullable=True),
        sa.Column('review_required', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('generated_by', sa.String(length=100), nullable=True),
        sa.Column('generation_status', sa.String(length=50), nullable=False, server_default='SUCCESS'),
        sa.Column('processing_time_ms', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # 12. processing_runs
    op.create_table(
        'processing_runs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('application_id', sa.String(length=100), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='STARTED'),
        sa.Column('current_agent', sa.String(length=50), nullable=False, server_default='agent_1'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('total_processing_time_ms', sa.Float(), nullable=False, server_default='0.0'),
        sa.ForeignKeyConstraint(['application_id'], ['loan_applications.application_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('processing_runs')
    op.drop_table('final_reports')
    op.drop_table('risk_factors')
    op.drop_table('risk_assessments')
    op.drop_table('cross_document_findings')
    op.drop_table('validation_results')
    op.drop_index(op.f('ix_extracted_fields_field_name'), table_name='extracted_fields')
    op.drop_table('extracted_fields')
    op.drop_table('document_classifications')
    op.drop_table('documents')
    op.drop_index(op.f('ix_document_requirements_loan_type'), table_name='document_requirements')
    op.drop_table('document_requirements')
    op.drop_table('applicants')
    op.drop_index(op.f('ix_loan_applications_application_id'), table_name='loan_applications')
    op.drop_table('loan_applications')
