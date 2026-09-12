"""
Shared state definitions for the Multi-Agent Loan Document Processing AI System.
This file contains shared Pydantic models and LangGraph State TypedDict structures
that allow seamless state passing across agents (Agent 1 -> Agent 2 -> ... Agent 6).
"""

from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field

from shared.policy import (
    DocumentRequirement,
    DocumentSlotStatus,
    ApplicationDocumentStatus,
    LOAN_DOCUMENT_POLICY,
    LOAN_TYPE_NAMES,
    get_loan_type_policy,
    is_document_acceptable_for_requirement
)



class DocumentContent(BaseModel):
    """
    Structured container for extracted document text, metadata, and quality metrics
    before classification.
    """
    file_name: str = Field(..., description="Original or sanitized filename")
    file_type: str = Field(..., description="File extension e.g. .pdf, .docx, .png")
    page_count: int = Field(default=1, ge=0)
    extracted_text: str = Field(..., description="Raw text extracted from document pages or OCR")
    ocr_text: Optional[str] = Field(default=None, description="OCR text if text extraction required OCR fallback")
    extraction_status: str = Field(default="success", description="Status: success, warning, failed")
    extraction_method: str = Field(..., description="Method used: pdf_text, docx_parser, txt_reader, image_ocr, pdf_ocr, etc.")


class DocumentClassificationResult(BaseModel):
    """
    Structured classification result produced by Agent 1.
    Serves as the explicit contract between Agent 1 and Agent 2 (Extraction Agent).
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Sanitized original filename")
    file_extension: str = Field(..., description="File extension including leading dot")
    file_path: Optional[str] = Field(default=None, description="Absolute file path on disk")
    document_type: str = Field(
        ...,
        description="One of: payslip, bank_statement, itr_tax_return, kyc_identity, employment_letter, form_16, address_proof, other, unknown"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence score between 0.0 and 1.0")
    classification_reason: str = Field(..., description="Explanation of why this document category was assigned")
    text_available: bool = Field(..., description="Whether usable text was extracted from the document")
    text_length: int = Field(..., ge=0, description="Length of extracted textual content in characters")
    page_count: int = Field(default=1, ge=0, description="Number of pages or 1 for single-page/image documents")
    extraction_method: str = Field(..., description="Method used to read content: pdf_text, docx_parser, txt_reader, image_ocr, svg_parser, etc.")
    content_quality: str = Field(default="good", description="Assessment of text quality: good, sparse, empty, corrupted")
    ocr_used: bool = Field(default=False, description="Whether OCR fallback was executed")
    ocr_success: bool = Field(default=True, description="Whether OCR executed successfully")
    normalized_document_type: Optional[str] = Field(default=None, description="Canonical normalized document type")
    processing_time_ms: float = Field(..., ge=0.0, description="Total processing time in milliseconds")
    status: str = Field(..., description="Processing status: success, warning, unknown, failed")
    error_type: Optional[str] = Field(default=None, description="Error category if processing or classification failed")
    next_agent: str = Field(default="extraction_agent", description="Contract field specifying the downstream agent")


class ExtractedField(BaseModel):
    """Represents a single extracted data field with value, confidence, and source evidence."""
    value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Optional[Dict[str, Any]] = None
    source_document: Optional[str] = None
    page_number: int = Field(default=1)
    evidence_text: Optional[str] = None
    evidence_id: Optional[str] = None
    extraction_method: Optional[str] = None
    ocr_used: bool = False


class DocumentExtractionResult(BaseModel):
    """
    Structured extraction result produced by Agent 2.
    Serves as the explicit contract between Agent 2 and Agent 3 (Validation Agent).
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Sanitized original filename")
    document_type: str = Field(..., description="Document type from Agent 1 classification")
    fields: Dict[str, ExtractedField] = Field(default_factory=dict, description="Extracted key-value fields with confidence and source evidence")
    overall_confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Aggregate extraction confidence score")
    extraction_status: str = Field(..., description="Status: success, warning, partial, failed")
    fields_requested: int = Field(default=0, description="Number of target fields requested for document type")
    fields_extracted: int = Field(default=0, description="Number of non-null fields successfully extracted")
    fields_missing: int = Field(default=0, description="Number of fields missing (null)")
    extraction_completeness: float = Field(default=0.0, ge=0.0, le=100.0, description="Completeness percentage: (fields_extracted / fields_requested) * 100")
    extraction_debug_info: Dict[str, Any] = Field(default_factory=dict, description="Development/debug metrics and execution breakdown")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Total extraction processing time in milliseconds")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Structured error records if extraction encountered issues")
    next_agent: str = Field(default="validation_agent", description="Contract field specifying the downstream agent")


class ValidationFinding(BaseModel):
    """Represents a single validation rule check result."""
    field_name: str = Field(..., description="Target field checked")
    status: str = Field(..., description="One of: PASS, WARNING, FAIL, NOT_CHECKED")
    finding_type: str = Field(default="OTHER_VALIDATION_ERROR", description="Classification of finding: MISSING_REQUIRED_DOCUMENT, MISSING_REQUIRED_FIELD, INVALID_VALUE, ARITHMETIC_ERROR, FORMAT_ERROR, READABILITY_ERROR, OTHER_VALIDATION_ERROR")
    message: str = Field(..., description="Human-readable message explaining validation result")
    severity: str = Field(default="INFO", description="One of: INFO, LOW, MEDIUM, HIGH")
    value: Any = Field(default=None, description="Field value that was validated")
    source: Optional[Dict[str, Any]] = Field(default=None, description="Source snippet/evidence if present")


class DocumentValidationResult(BaseModel):
    """
    Structured validation result produced by Agent 3.
    Serves as the explicit contract between Agent 3 and Agent 4 (Cross-Document Agent).
    """
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Sanitized original filename")
    document_type: str = Field(..., description="Document type from classification/extraction")
    fields: Dict[str, Any] = Field(default_factory=dict, description="Extracted key-value fields carried over from Agent 2")
    validation_status: str = Field(..., description="One of: PASS, WARNING, FAIL, NOT_APPLICABLE")
    validation_score: float = Field(default=100.0, ge=0.0, le=100.0, description="Deterministic validation quality score (0-100)")
    required_fields_present: int = Field(default=0, description="Count of required fields present and non-null")
    total_fields: int = Field(default=0, description="Total fields evaluated for this document schema")
    valid_fields: int = Field(default=0, description="Count of fields passing format, range, and type validation")
    warning_count: int = Field(default=0, description="Count of non-critical warning findings")
    error_count: int = Field(default=0, description="Count of critical failure findings")
    findings: List[ValidationFinding] = Field(default_factory=list, description="List of individual rule validation findings")
    field_validation: Dict[str, Any] = Field(default_factory=dict, description="Summary map of field validation statuses")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Total validation processing time in milliseconds")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error records if validation encountered unexpected exceptions")
    next_agent: str = Field(default="cross_document_agent", description="Contract field specifying downstream agent")


class CrossDocumentFinding(BaseModel):
    """Represents a single cross-document verification comparison result."""
    field_name: str = Field(..., description="Target comparison field (e.g. applicant_name, pan_number, monthly_income)")
    doc1_type: str = Field(..., description="Document type of first source")
    doc1_value: Any = Field(default=None, description="Extracted value from first source")
    doc2_type: str = Field(..., description="Document type of second source")
    doc2_value: Any = Field(default=None, description="Extracted value from second source")
    comparison_result: str = Field(..., description="One of: MATCH, MINOR_VARIATION, MISMATCH, UNVERIFIABLE, NOT_APPLICABLE")
    severity: str = Field(default="INFO", description="One of: INFO, LOW, MEDIUM, HIGH, CRITICAL")
    message: str = Field(..., description="Human-readable explanation of comparison")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score of comparison")
    source: Optional[Dict[str, Any]] = Field(default=None, description="Source context or details")


class CrossDocumentResult(BaseModel):
    """
    Structured cross-document verification result produced by Agent 4.
    Serves as the explicit contract between Agent 4 and Agent 5 (Risk Anomaly Agent).
    """
    application_id: str = Field(..., description="Unique application session ID")
    loan_type: str = Field(..., description="Loan type policy under which documents were evaluated")
    consistency_score: Optional[float] = Field(default=None, description="Overall cross-document consistency score (0-100 or null if insufficient evidence)")
    verification_coverage: float = Field(default=0.0, ge=0.0, le=100.0, description="Percentage of comparisons that could be verified (0-100%)")
    verification_status: str = Field(default="PASS", description="One of: PASS, WARNING, FAIL, INSUFFICIENT_EVIDENCE")
    total_comparisons: int = Field(default=0, description="Total field comparisons executed")
    verified_comparisons: int = Field(default=0, description="Number of comparisons with valid data (match + minor_variation + mismatch)")
    match_count: int = Field(default=0, description="Number of exact/normalized matches")
    minor_variation_count: int = Field(default=0, description="Number of minor variations (e.g. address/name abbreviation)")
    mismatch_count: int = Field(default=0, description="Number of conflicting mismatches")
    unverifiable_count: int = Field(default=0, description="Number of comparisons skipped due to missing/null values")
    not_applicable_count: int = Field(default=0, description="Number of comparisons skipped due to role separation")
    findings: List[CrossDocumentFinding] = Field(default_factory=list, description="List of individual cross-document findings")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Total Agent 4 execution time in ms")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error records if verification encountered exceptions")
    next_agent: str = Field(default="risk_anomaly_agent", description="Contract field specifying downstream agent")


class RiskFactor(BaseModel):
    """
    Structured risk factor item produced by Agent 5.
    Represents a specific anomaly, mismatch, missing document, or data quality risk.
    """
    factor_id: str = Field(..., description="Unique identifier for this risk factor")
    category: str = Field(..., description="Risk category: IDENTITY_RISK, INCOME_RISK, EMPLOYMENT_RISK, BANKING_RISK, DOCUMENT_RISK, CROSS_DOCUMENT_RISK, PROPERTY_RISK, VEHICLE_RISK, EDUCATION_RISK, BUSINESS_RISK, GOLD_RISK, AGRICULTURE_RISK, FD_RISK, FINANCIAL_RISK, DATA_QUALITY_RISK, OTHER")
    severity: str = Field(..., description="One of: LOW, MEDIUM, HIGH, CRITICAL")
    title: str = Field(..., description="Short title describing the risk factor")
    description: str = Field(..., description="Detailed explanation of the risk factor")
    source: str = Field(default="Agent 5 Risk Engine", description="Source agent or module that detected this risk")
    document_a: Optional[str] = Field(default=None, description="First source document name/type involved")
    document_b: Optional[str] = Field(default=None, description="Second source document name/type involved")
    field: Optional[str] = Field(default=None, description="Target field name involved")
    value_a: Any = Field(default=None, description="First value")
    value_b: Any = Field(default=None, description="Second value")
    evidence: Optional[str] = Field(default=None, description="Concrete snippet or rationale evidence")
    points: float = Field(default=0.0, ge=0.0, description="Risk impact points assigned to this factor")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    recommendation: Optional[str] = Field(default=None, description="Actionable recommendation for underwriter/reviewer")


class RiskAssessmentResult(BaseModel):
    """
    Structured risk assessment result produced by Agent 5.
    Serves as the explicit contract between Agent 5 and Agent 6 (Final Decision Agent).
    """
    application_id: str = Field(..., description="Unique application session ID")
    loan_type: str = Field(..., description="Loan type evaluated")
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0, description="Overall risk score (0-100, capped at 100)")
    risk_level: str = Field(default="LOW", description="One of: LOW (0-24), MEDIUM (25-49), HIGH (50-74), CRITICAL (75-100)")
    total_risk_factors: int = Field(default=0, description="Total count of active risk factors detected")
    critical_count: int = Field(default=0, description="Count of CRITICAL risk factors")
    high_count: int = Field(default=0, description="Count of HIGH risk factors")
    medium_count: int = Field(default=0, description="Count of MEDIUM risk factors")
    low_count: int = Field(default=0, description="Count of LOW risk factors")
    risk_factors: List[RiskFactor] = Field(default_factory=list, description="Detailed list of risk factors")
    anomalies: List[Dict[str, Any]] = Field(default_factory=list, description="List of detected structural or semantic anomalies")
    overall_summary: str = Field(default="", description="Executive risk summary text")
    recommended_action: str = Field(default="PROCEED_TO_REPORT", description="One of: PROCEED_TO_REPORT, REVIEW_RECOMMENDED, HUMAN_REVIEW_REQUIRED")
    verification_coverage: float = Field(default=0.0, ge=0.0, le=100.0, description="Verification coverage carried over from Agent 4")
    consistency_score: Optional[float] = Field(default=None, description="Consistency score carried over from Agent 4")
    semantic_reasoning_status: str = Field(default="UNAVAILABLE", description="One of: AVAILABLE, UNAVAILABLE, MOCKED_TEST, REAL_OLLAMA_TEST")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Agent 5 processing time in milliseconds")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Error logs encountered during risk assessment")
    next_agent: str = Field(default="final_report_agent", description="Contract field specifying downstream agent")


class LoanDocumentState(TypedDict, total=False):
    """
    Shared LangGraph State TypedDict passed through the multi-agent pipeline.
    Agent 1 populates classification_results -> next_agent = extraction_agent
    Agent 2 populates extraction_results -> next_agent = validation_agent
    Agent 3 populates validation_results -> next_agent = cross_document_agent
    Agent 4 populates cross_document_results -> next_agent = risk_anomaly_agent
    Agent 5 populates risk_assessment_results -> next_agent = final_report_agent
    """
    application_id: str
    loan_type: Optional[str]
    documents: List[Dict[str, Any]]
    current_document: Optional[Dict[str, Any]]
    classification_results: List[Dict[str, Any]]
    extraction_results: List[Dict[str, Any]]
    validation_results: List[Dict[str, Any]]
    cross_document_results: Optional[Dict[str, Any]]
    risk_assessment_results: Optional[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    processing_metrics: Dict[str, Any]
    next_agent: str
    doc_risks: List[Any]
    val_risks: List[Any]
    cross_risks: List[Any]
    domain_risks: List[Any]
    risk_score: float
    risk_level: str
    total_risk_factors: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    risk_factors: List[Any]
    anomalies: List[Any]
    overall_summary: str
    recommended_action: str
    verification_coverage: float
    consistency_score: Optional[float]
    semantic_reasoning_status: str
    _has_classification_input: bool
    _doc_summary: Any
    _val_summary: Any
    _cross_summary: Any
    _risk_summary: Any
    _deterministic_decision: str
    _decision_reason: str
    _genai_summary: str
    final_report: Optional[Dict[str, Any]]


class DocumentSummary(BaseModel):
    """Document completeness breakdown."""
    uploaded_count: int = Field(default=0, ge=0)
    required_count: int = Field(default=0, ge=0)
    satisfied_count: int = Field(default=0, ge=0)
    missing_count: int = Field(default=0, ge=0)
    wrong_count: int = Field(default=0, ge=0)
    optional_count: int = Field(default=0, ge=0)
    status: str = Field(default="INCOMPLETE", description="One of: COMPLETE, INCOMPLETE, WRONG_DOCUMENT")


class ValidationSummary(BaseModel):
    """Validation checks breakdown from Agent 3."""
    total_checks: int = Field(default=0, ge=0)
    passed_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    blocking_count: int = Field(default=0, ge=0)


class CrossDocumentSummary(BaseModel):
    """Cross-document verification breakdown from Agent 4."""
    verification_coverage: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: Optional[float] = Field(default=None)
    total_comparisons: int = Field(default=0, ge=0)
    match_count: int = Field(default=0, ge=0)
    minor_variation_count: int = Field(default=0, ge=0)
    mismatch_count: int = Field(default=0, ge=0)
    unverifiable_count: int = Field(default=0, ge=0)


class RiskSummary(BaseModel):
    """Risk breakdown from Agent 5."""
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: str = Field(default="LOW", description="One of: LOW, MEDIUM, HIGH, CRITICAL")
    critical_count: int = Field(default=0, ge=0)
    high_count: int = Field(default=0, ge=0)
    medium_count: int = Field(default=0, ge=0)
    low_count: int = Field(default=0, ge=0)


class FinalReport(BaseModel):
    """
    Structured final loan application report produced by Agent 6.
    Consumes outputs from Agents 1–5 to produce final decision and GenAI executive summary.
    """
    application_id: str = Field(..., description="Unique application session ID")
    loan_type: str = Field(..., description="Target loan policy type")
    decision: str = Field(..., description="One of: PASS, HUMAN_REVIEW, INSUFFICIENT_EVIDENCE")
    decision_reason: str = Field(..., description="Primary reason for the decision")
    risk_score: float = Field(default=0.0, ge=0.0, le=100.0)
    risk_level: str = Field(default="LOW")
    verification_coverage: float = Field(default=0.0, ge=0.0, le=100.0)
    consistency_score: Optional[float] = Field(default=None)
    document_summary: DocumentSummary = Field(default_factory=DocumentSummary)
    validation_summary: ValidationSummary = Field(default_factory=ValidationSummary)
    cross_document_summary: CrossDocumentSummary = Field(default_factory=CrossDocumentSummary)
    risk_summary: RiskSummary = Field(default_factory=RiskSummary)
    key_findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    executive_summary: str = Field(..., description="3-5 sentence GenAI or deterministic executive summary")
    review_required: bool = Field(default=False)
    generated_by: str = Field(default="Agent 6 — Final Report Engine")
    generation_status: str = Field(default="GENAI_GENERATED", description="One of: GENAI_GENERATED, DETERMINISTIC_FALLBACK, ERROR")
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    agent_runtimes: Dict[str, float] = Field(default_factory=dict)
    eligibility_decision: Optional[Dict[str, Any]] = Field(default=None)
    policy_citation_count: int = Field(default=0)
    field_evidence_count: int = Field(default=0)
    decision_graph_id: Optional[str] = Field(default=None)
    telemetry_summary: Optional[Dict[str, Any]] = Field(default=None)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    next_agent: str = Field(default="completed")



LOAN_TYPE_EXPECTED_DOCUMENTS: Dict[str, List[str]] = {
    "home_loan": [
        "kyc_identity", "pan_card", "payslip", "itr_tax_return", "bank_statement",
        "property_title_document", "sale_deed", "sale_agreement", "approved_building_plan"
    ],
    "personal_loan": [
        "kyc_identity", "pan_card", "payslip", "bank_statement", "employment_letter"
    ],
    "vehicle_loan": [
        "kyc_identity", "pan_card", "payslip", "itr_tax_return", "bank_statement",
        "vehicle_quotation", "vehicle_invoice"
    ],
    "education_loan": [
        "student_kyc", "admission_letter", "fee_structure", "academic_certificate",
        "co_applicant_kyc", "co_applicant_income_proof"
    ],
    "business_loan": [
        "kyc_identity", "business_registration", "gst_certificate", "gst_return",
        "business_itr", "business_bank_statement", "profit_loss_statement", "balance_sheet"
    ],
    "gold_loan": [
        "kyc_identity", "gold_security_document", "jewellery_valuation_report", "pledge_document"
    ],
    "loan_against_property": [
        "kyc_identity", "payslip", "itr_tax_return", "property_title_document",
        "property_tax_receipt", "property_valuation_report"
    ],
    "agriculture_loan": [
        "kyc_identity", "land_record", "cultivation_record", "agricultural_income_proof", "bank_statement"
    ],
    "loan_against_fd": [
        "kyc_identity", "fixed_deposit_certificate", "fixed_deposit_receipt", "fd_statement"
    ],
    "consumer_durable_loan": [
        "kyc_identity", "payslip", "bank_statement", "product_quotation", "product_invoice"
    ]
}
