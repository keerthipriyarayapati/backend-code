"""
Field-Level and Decision Evidence Citations Module.
Provides provenance tracking, snippet extraction, and PII masking.
"""

from evidence.schemas import FieldEvidence, DecisionCitation, DecisionEvidence
from evidence.manager import (
    mask_pii_value,
    create_field_evidence,
    build_rule_citations,
    save_application_evidence,
    get_application_evidence
)

__all__ = [
    "FieldEvidence",
    "DecisionCitation",
    "DecisionEvidence",
    "mask_pii_value",
    "create_field_evidence",
    "build_rule_citations",
    "save_application_evidence",
    "get_application_evidence"
]
