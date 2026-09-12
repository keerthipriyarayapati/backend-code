"""
Pydantic Schemas for Field-Level and Decision Evidence Citations.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class FieldEvidence(BaseModel):
    """Field-level extraction citation with document provenance."""
    evidence_id: str = Field(..., description="Unique evidence identifier e.g. EV_pan_card_pan_number_01")
    application_id: str = Field(..., description="Associated loan application ID")
    document_id: Optional[int] = Field(None, description="Source document database ID")
    slot_id: Optional[str] = Field(None, description="Document requirement slot ID")
    document_type: Optional[str] = Field(None, description="Classified canonical document type")
    field_name: str = Field(..., description="Extracted field name")
    extracted_value: Optional[str] = Field(None, description="Masked extracted value")
    raw_value: Optional[str] = Field(None, description="Raw extracted string")
    normalized_value: Optional[str] = Field(None, description="Cleaned/normalized value")
    source_page: int = Field(default=1, description="Page number where snippet was located")
    snippet: Optional[str] = Field(None, description="Source text snippet containing extracted value")
    bounding_box: Optional[Dict[str, Any]] = Field(None, description="Coordinates bounding box if OCR/visual")
    extraction_method: str = Field(default="Regex / Key-Value Extractor", description="Extraction technique applied")
    ocr_used: bool = Field(default=False, description="Whether Tesseract OCR was invoked")
    confidence: float = Field(default=1.0, description="Extraction confidence score (0.0 to 1.0)")
    validation_status: str = Field(default="VALIDATED", description="Field validation status")
    created_at: Optional[str] = Field(None, description="ISO timestamp")


class DecisionCitation(BaseModel):
    """Individual evidence citation linked to a policy or eligibility rule."""
    citation_id: str = Field(..., description="Unique citation identifier")
    citation_type: str = Field(..., description="FIELD, POLICY_RULE, CROSS_DOC, RISK, or DOCUMENT")
    source_title: str = Field(..., description="Human-readable title of the source (e.g. 'PAN Card (Page 1)')")
    reference_id: str = Field(..., description="Referenced entity ID (evidence_id, rule_id, etc.)")
    snippet: Optional[str] = Field(None, description="Verbatim or summarized evidence snippet")
    confidence: float = Field(default=1.0, description="Citation reliability confidence")
    masked_summary: Optional[str] = Field(None, description="Safe masked summary for display")


class DecisionEvidence(BaseModel):
    """Complete multi-citation evidence bundle for an eligibility or underwriting decision."""
    decision_id: str = Field(..., description="Unique decision outcome ID")
    rule_code: str = Field(..., description="Policy rule evaluated")
    decision_type: str = Field(..., description="ELIGIBILITY, UNDERWRITING, or COMPLIANCE")
    status: str = Field(..., description="PASS, FAIL, INSUFFICIENT_EVIDENCE, or SKIPPED")
    citations: List[DecisionCitation] = Field(default_factory=list, description="Citations supporting this decision")
    explanation: Optional[str] = Field(None, description="Human-readable decision explanation citing evidence")
