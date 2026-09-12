"""
Pydantic Schemas for Loan Eligibility Engine.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class RuleEvaluationResult(BaseModel):
    """Result of evaluating a single underwriting policy rule."""
    rule_code: str = Field(..., description="Unique policy rule code")
    category: str = Field(..., description="Rule category e.g. INCOME, AGE, CREDIT, COLLATERAL")
    field_name: Optional[str] = Field(None, description="Field name evaluated")
    operator: str = Field(..., description="Evaluation operator: GTE, LTE, EQ, IN, EXISTS, MATCHES")
    expected_value: Optional[str] = Field(None, description="Expected value or threshold")
    actual_value: Optional[str] = Field(None, description="Actual extracted and masked value")
    status: str = Field(..., description="PASS, FAIL, SKIPPED, or INSUFFICIENT_EVIDENCE")
    mandatory: bool = Field(default=True, description="Whether rule failure blocks approval")
    severity: str = Field(default="CRITICAL", description="CRITICAL, HIGH, MEDIUM, LOW")
    failure_reason: Optional[str] = Field(None, description="Explanation if status is not PASS")
    evidence_id: Optional[str] = Field(None, description="Associated FieldEvidence ID")
    policy_rule_id: Optional[str] = Field(None, description="Associated PolicyRule ID")


class EligibilityDecision(BaseModel):
    """Overall loan eligibility determination with full decision audit trail."""
    eligibility_id: str = Field(..., description="Unique eligibility determination ID")
    application_id: str = Field(..., description="Loan application ID")
    loan_type: str = Field(..., description="Loan type assessed")
    decision: str = Field(..., description="ELIGIBLE, NOT_ELIGIBLE, HUMAN_REVIEW_REQUIRED, INSUFFICIENT_EVIDENCE")
    confidence: float = Field(default=1.0, description="Decision confidence score (0.0 to 1.0)")
    reasons: List[str] = Field(default_factory=list, description="Ordered human-readable reasons for decision")
    rules_evaluated_count: int = Field(default=0)
    rules_passed_count: int = Field(default=0)
    rules_failed_count: int = Field(default=0)
    rules_skipped_count: int = Field(default=0)
    processing_time_ms: float = Field(default=0.0)
    rule_results: List[RuleEvaluationResult] = Field(default_factory=list)
