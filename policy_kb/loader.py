"""
Policy Knowledge Base Loader & Database Synchronization.
Seeds and retrieves loan policies, rules, and document requirements.
"""

import logging
from typing import Optional, Dict, Any, List
from sqlalchemy.orm import Session

from policy_kb.definitions import ALL_LOAN_POLICIES, get_policy_definition
from database.repositories import (
    upsert_loan_policy,
    get_policy_by_type,
    list_all_policies as repo_list_all_policies
)

logger = logging.getLogger("PolicyKBLoader")


def seed_loan_policies(db: Session) -> int:
    """Seeds or refreshes all 10 loan underwriting policies into the database."""
    count = 0
    for pol in ALL_LOAN_POLICIES:
        policy_data = {
            "policy_id": pol["policy_id"],
            "loan_type": pol["loan_type"],
            "policy_name": pol["policy_name"],
            "version": pol["version"],
            "description": pol["description"],
            "effective_date": pol["effective_date"],
            "source_type": pol["source_type"],
            "source_document": pol["source_document"],
            "source_section": pol["source_section"],
            "source_page": pol["source_page"],
            "status": pol["status"]
        }
        
        rules_data = pol.get("rules", [])
        required_docs = pol.get("required_documents", [])
        # Enrich required_docs with loan_type
        for rd in required_docs:
            rd["loan_type"] = pol["loan_type"]
            
        required_fields = pol.get("required_fields", [])
        for rf in required_fields:
            rf["loan_type"] = pol["loan_type"]

        upsert_loan_policy(
            db=db,
            policy_data=policy_data,
            rules_data=rules_data,
            required_docs_data=required_docs,
            required_fields_data=required_fields
        )
        count += 1
        
    logger.info(f"Successfully seeded {count} loan underwriting policies.")
    return count


def get_active_policy(db: Session, loan_type: str) -> Optional[Dict[str, Any]]:
    """Retrieves policy definition from DB, with fallback to hardcoded definitions."""
    db_policy = get_policy_by_type(db, loan_type)
    if db_policy:
        return {
            "policy_id": db_policy.policy_id,
            "loan_type": db_policy.loan_type,
            "policy_name": db_policy.policy_name,
            "version": db_policy.version,
            "description": db_policy.description,
            "effective_date": db_policy.effective_date,
            "source_type": db_policy.source_type,
            "source_document": db_policy.source_document,
            "source_section": db_policy.source_section,
            "source_page": db_policy.source_page,
            "status": db_policy.status,
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "rule_code": r.rule_code,
                    "category": r.category,
                    "field_name": r.field_name,
                    "operator": r.operator,
                    "expected_value": r.expected_value,
                    "threshold_value": r.threshold_value,
                    "severity": r.severity,
                    "mandatory": r.mandatory,
                    "error_message": r.error_message,
                    "source_type": r.source_type,
                    "source_document": r.source_document,
                    "source_section": r.source_section,
                    "source_page": r.source_page
                } for r in db_policy.rules
            ],
            "required_documents": [
                {
                    "slot_id": d.slot_id,
                    "document_type": d.document_type,
                    "display_name": d.display_name,
                    "required": d.required
                } for d in db_policy.required_documents
            ]
        }
    
    # Fallback if DB not seeded yet
    return get_policy_definition(loan_type)
