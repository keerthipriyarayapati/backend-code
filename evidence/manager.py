"""
Evidence Manager & Citation Tracking Engine.
Handles evidence collection, snippet extraction, PII masking, and multi-source citation linking.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from evidence.schemas import FieldEvidence, DecisionCitation, DecisionEvidence
from agents.agent_5_risk.agent import mask_sensitive_value
from database.repositories import (
    save_field_evidence_batch,
    get_field_evidence_by_application,
    get_evidence_by_id
)

logger = logging.getLogger("EvidenceManager")


def mask_pii_value(val: Any, field_name: str = "") -> str:
    """Masks sensitive values (PAN, Aadhaar, Bank Account, FD, Chassis, Engine, Phone)."""
    return mask_sensitive_value(val, field_name)


def create_field_evidence(
    application_id: str,
    field_name: str,
    raw_value: Any,
    document_id: Optional[int] = None,
    slot_id: Optional[str] = None,
    document_type: Optional[str] = None,
    source_page: int = 1,
    snippet: Optional[str] = None,
    bounding_box: Optional[Dict[str, Any]] = None,
    extraction_method: str = "Regex / Key-Value Extractor",
    ocr_used: bool = False,
    confidence: float = 1.0,
    validation_status: str = "VALIDATED"
) -> FieldEvidence:
    """Creates a structured FieldEvidence citation object with PII masking."""
    raw_str = str(raw_value) if raw_value is not None else ""
    masked_val = mask_pii_value(raw_str, field_name) if raw_str else ""
    
    # Mask snippet if it contains the raw sensitive value
    safe_snippet = snippet or ""
    if raw_str and masked_val != raw_str and raw_str in safe_snippet:
        safe_snippet = safe_snippet.replace(raw_str, masked_val)

    # Deterministic or unique evidence ID
    clean_field = field_name.replace(" ", "_").lower()
    ev_suffix = uuid.uuid4().hex[:6]
    evidence_id = f"EV_{application_id}_{clean_field}_{ev_suffix}"

    return FieldEvidence(
        evidence_id=evidence_id,
        application_id=application_id,
        document_id=document_id,
        slot_id=slot_id,
        document_type=document_type,
        field_name=field_name,
        extracted_value=masked_val,
        raw_value=masked_val,  # Persist masked value for audit safety
        normalized_value=raw_str.lower().strip() if raw_str else "",
        source_page=source_page,
        snippet=safe_snippet,
        bounding_box=bounding_box,
        extraction_method=extraction_method,
        ocr_used=ocr_used,
        confidence=confidence,
        validation_status=validation_status
    )


def build_rule_citations(
    rule_code: str,
    rule_definition: Dict[str, Any],
    field_evidence: Optional[FieldEvidence] = None,
    extra_citations: Optional[List[DecisionCitation]] = None
) -> List[DecisionCitation]:
    """Builds a rich citation list for an evaluated eligibility rule."""
    citations: List[DecisionCitation] = []

    # 1. Policy Rule Citation
    citations.append(
        DecisionCitation(
            citation_id=f"CIT_POL_{rule_code}_{uuid.uuid4().hex[:4]}",
            citation_type="POLICY_RULE",
            source_title=f"{rule_definition.get('source_document', 'Credit Policy')} ({rule_definition.get('source_section', 'Underwriting')}, Page {rule_definition.get('source_page', 'N/A')})",
            reference_id=rule_definition.get("rule_id", rule_code),
            snippet=rule_definition.get("error_message") or f"Rule {rule_code}: {rule_definition.get('operator')} {rule_definition.get('expected_value')}",
            confidence=1.0,
            masked_summary=f"Policy requirement: {rule_code}"
        )
    )

    # 2. Document Field Evidence Citation
    if field_evidence:
        citations.append(
            DecisionCitation(
                citation_id=f"CIT_EV_{field_evidence.field_name}_{uuid.uuid4().hex[:4]}",
                citation_type="FIELD",
                source_title=f"{field_evidence.document_type or 'Document'} (Page {field_evidence.source_page})",
                reference_id=field_evidence.evidence_id,
                snippet=field_evidence.snippet or f"Extracted {field_evidence.field_name}: {field_evidence.extracted_value}",
                confidence=field_evidence.confidence,
                masked_summary=f"{field_evidence.field_name} = {field_evidence.extracted_value}"
            )
        )

    # 3. Any additional citations (e.g. cross-doc, risk)
    if extra_citations:
        citations.extend(extra_citations)

    return citations


def save_application_evidence(
    db: Session,
    application_id: str,
    evidence_list: List[FieldEvidence]
) -> int:
    """Persists a list of FieldEvidence items to the database."""
    items = [ev.dict() for ev in evidence_list]
    saved = save_field_evidence_batch(db, items)
    logger.info(f"Persisted {len(saved)} field evidence citations for {application_id}")
    return len(saved)


def get_application_evidence(db: Session, application_id: str) -> List[Dict[str, Any]]:
    """Retrieves all field evidence citations for an application."""
    return get_field_evidence_by_application(db, application_id)
