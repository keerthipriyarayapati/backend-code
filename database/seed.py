"""
Database Seed Engine for Loan Document Requirements.
Populates document_requirements table for all 10 loan types from central LOAN_DOCUMENT_POLICY.
"""

import logging
from sqlalchemy.orm import Session

from database.models import DocumentRequirementModel
from shared.policy import LOAN_DOCUMENT_POLICY, LOAN_TYPE_NAMES

logger = logging.getLogger("DatabaseSeed")


def seed_document_requirements(db: Session):
    """
    Populates document_requirements table for all 10 loan types.
    Serves as the database source of truth for loan document checklists.
    """
    logger.info("Verifying and seeding document requirements for all 10 loan types...")
    total_added = 0

    for loan_type, policy in LOAN_DOCUMENT_POLICY.items():
        existing = db.query(DocumentRequirementModel).filter(DocumentRequirementModel.loan_type == loan_type).all()
        policy_reqs = policy.get("required", []) + policy.get("optional", [])
        policy_ids = {r.requirement_id for r in policy_reqs}
        existing_ids = {r.document_type for r in existing}

        if policy_ids != existing_ids:
            db.query(DocumentRequirementModel).filter(DocumentRequirementModel.loan_type == loan_type).delete()
            db.commit()

            seeded_records = []
            for req in policy.get("required", []):
                rec = DocumentRequirementModel(
                    loan_type=loan_type,
                    document_type=req.requirement_id,
                    display_name=req.display_name,
                    purpose=f"Mandatory requirement for {LOAN_TYPE_NAMES.get(loan_type, loan_type)}",
                    requirement_status="REQUIRED",
                    applicant_role=req.entity_role or "PRIMARY",
                    allowed_extensions=".pdf,.docx,.jpg,.jpeg,.png,.txt",
                    description=req.description
                )
                seeded_records.append(rec)

            for opt in policy.get("optional", []):
                rec = DocumentRequirementModel(
                    loan_type=loan_type,
                    document_type=opt.requirement_id,
                    display_name=opt.display_name,
                    purpose=f"Optional supporting requirement for {LOAN_TYPE_NAMES.get(loan_type, loan_type)}",
                    requirement_status="OPTIONAL",
                    applicant_role=opt.entity_role or "PRIMARY",
                    allowed_extensions=".pdf,.docx,.jpg,.jpeg,.png,.txt",
                    description=opt.description
                )
                seeded_records.append(rec)

            db.add_all(seeded_records)
            db.commit()
            total_added += len(seeded_records)

    logger.info(f"Successfully verified document requirements in SQLite (synced {total_added} records).")
