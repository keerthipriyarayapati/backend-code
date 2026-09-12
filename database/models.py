"""
SQLAlchemy ORM Models for Loan Document Processing AI System.
Defines the 12 core tables required for full application persistence, audit trail, and document metadata tracking.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, DateTime, ForeignKey
)
from sqlalchemy.orm import relationship
from database.connection import Base


class LoanApplication(Base):
    """TABLE 1: loan_applications"""
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), unique=True, index=True, nullable=False)
    loan_type = Column(String(100), nullable=False)
    applicant_name = Column(String(200), nullable=True)
    status = Column(String(50), default="NOT_STARTED", nullable=False)
    employee_id = Column(String(100), nullable=True, index=True)
    branch_id = Column(String(100), nullable=True, index=True)
    risk_level = Column(String(50), nullable=True)
    completed_at = Column(DateTime, nullable=True)
    processing_time = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    applicants = relationship("Applicant", back_populates="application", cascade="all, delete-orphan")
    documents = relationship("DocumentModel", back_populates="application", cascade="all, delete-orphan")
    validation_results = relationship("ValidationResultModel", back_populates="application", cascade="all, delete-orphan")
    cross_document_findings = relationship("CrossDocumentFindingModel", back_populates="application", cascade="all, delete-orphan")
    risk_assessments = relationship("RiskAssessmentModel", back_populates="application", cascade="all, delete-orphan")
    final_reports = relationship("FinalReportModel", back_populates="application", cascade="all, delete-orphan")
    processing_runs = relationship("ProcessingRunModel", back_populates="application", cascade="all, delete-orphan")
    field_evidences = relationship("FieldEvidenceModel", back_populates="application", cascade="all, delete-orphan")
    decision_graphs = relationship("DecisionGraphModel", back_populates="application", cascade="all, delete-orphan")
    eligibility_results = relationship("EligibilityResultModel", back_populates="application", cascade="all, delete-orphan")
    agent_execution_logs = relationship("AgentExecutionLogModel", back_populates="application", cascade="all, delete-orphan")
    telemetry_metrics = relationship("TelemetryMetricsModel", back_populates="application", cascade="all, delete-orphan")


class Applicant(Base):
    """TABLE 2: applicants"""
    __tablename__ = "applicants"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    applicant_type = Column(String(50), default="PRIMARY_APPLICANT", nullable=False)
    full_name = Column(String(200), nullable=True)
    date_of_birth = Column(String(50), nullable=True)
    gender = Column(String(20), nullable=True)
    email = Column(String(150), nullable=True)
    phone = Column(String(50), nullable=True)
    address = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="applicants")
    documents = relationship("DocumentModel", back_populates="applicant")


class DocumentRequirementModel(Base):
    """TABLE 3: document_requirements"""
    __tablename__ = "document_requirements"

    id = Column(Integer, primary_key=True, autoincrement=True)
    loan_type = Column(String(100), index=True, nullable=False)
    document_type = Column(String(100), nullable=False)  # requirement_id slot identifier
    display_name = Column(String(200), nullable=False)
    purpose = Column(Text, nullable=True)
    requirement_status = Column(String(50), default="REQUIRED", nullable=False)  # REQUIRED, OPTIONAL, SUPPORTING, NOT_APPLICABLE
    applicant_role = Column(String(50), default="PRIMARY", nullable=False)
    allowed_extensions = Column(String(200), default=".pdf,.docx,.jpg,.jpeg,.png,.txt", nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DocumentModel(Base):
    """TABLE 4: documents"""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    applicant_id = Column(Integer, ForeignKey("applicants.id", ondelete="SET NULL"), nullable=True)
    requirement_id = Column(String(100), nullable=True)
    file_name = Column(String(255), nullable=False)
    file_path = Column(Text, nullable=False)
    file_extension = Column(String(20), nullable=True)
    mime_type = Column(String(100), nullable=True)
    document_type = Column(String(100), nullable=True)  # Classification output from Agent 1
    upload_status = Column(String(50), default="accepted", nullable=False)  # accepted, pending, wrong_document, duplicate, error
    extraction_method = Column(String(100), nullable=True)
    ocr_used = Column(Boolean, default=False, nullable=False)
    ocr_success = Column(Boolean, default=False, nullable=False)
    extraction_error = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    text_quality = Column(String(50), nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="documents")
    applicant = relationship("Applicant", back_populates="documents")
    classifications = relationship("DocumentClassificationModel", back_populates="document", cascade="all, delete-orphan")
    extracted_fields = relationship("ExtractedFieldModel", back_populates="document", cascade="all, delete-orphan")


class DocumentClassificationModel(Base):
    """TABLE 5: document_classifications (Agent 1 Output)"""
    __tablename__ = "document_classifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    predicted_document_type = Column(String(100), nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    classification_status = Column(String(50), default="CLASSIFIED", nullable=False)  # CLASSIFIED, UNKNOWN, LOW_CONFIDENCE, WRONG_DOCUMENT
    loan_type = Column(String(100), nullable=True)
    classification_reason = Column(Text, nullable=True)
    model_name = Column(String(100), default="Agent1_DocumentClassifier", nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("DocumentModel", back_populates="classifications")


class ExtractedFieldModel(Base):
    """TABLE 6: extracted_fields (Agent 2 Output)"""
    __tablename__ = "extracted_fields"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    field_name = Column(String(100), index=True, nullable=False)
    field_value = Column(Text, nullable=True)
    normalized_value = Column(Text, nullable=True)
    data_type = Column(String(50), default="string", nullable=False)
    confidence = Column(Float, default=0.0, nullable=False)
    source_page = Column(Integer, default=1, nullable=False)
    source_text = Column(Text, nullable=True)
    extraction_method = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    document = relationship("DocumentModel", back_populates="extracted_fields")


class ValidationResultModel(Base):
    """TABLE 7: validation_results (Agent 3 Output)"""
    __tablename__ = "validation_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    field_name = Column(String(100), nullable=False)
    validation_type = Column(String(100), nullable=True)
    status = Column(String(50), default="PASS", nullable=False)  # PASS, WARNING, FAIL, NOT_APPLICABLE
    severity = Column(String(50), default="INFO", nullable=False)  # INFO, LOW, MEDIUM, HIGH
    expected_value = Column(Text, nullable=True)
    actual_value = Column(Text, nullable=True)
    explanation = Column(Text, nullable=True)
    evidence = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="validation_results")


class CrossDocumentFindingModel(Base):
    """TABLE 8: cross_document_findings (Agent 4 Output)"""
    __tablename__ = "cross_document_findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    document_a_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    document_b_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    field = Column(String(100), nullable=False)
    category = Column(String(100), nullable=True)
    severity = Column(String(50), default="INFO", nullable=False)
    value_a = Column(Text, nullable=True)
    value_b = Column(Text, nullable=True)
    normalized_value_a = Column(Text, nullable=True)
    normalized_value_b = Column(Text, nullable=True)
    comparison_result = Column(String(50), nullable=False)  # MATCH, MINOR_VARIATION, MISMATCH, UNVERIFIABLE, NOT_APPLICABLE
    explanation = Column(Text, nullable=True)
    confidence = Column(Float, default=1.0, nullable=False)
    evidence = Column(Text, nullable=True)
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="cross_document_findings")


class RiskAssessmentModel(Base):
    """TABLE 9: risk_assessments (Agent 5 Output Header)"""
    __tablename__ = "risk_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    risk_score = Column(Float, default=0.0, nullable=False)
    risk_level = Column(String(50), default="LOW", nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    recommended_action = Column(String(100), default="PROCEED_TO_REPORT", nullable=False)  # PROCEED_TO_REPORT, REVIEW_RECOMMENDED, HUMAN_REVIEW_REQUIRED
    overall_reason = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="risk_assessments")
    risk_factors = relationship("RiskFactorModel", back_populates="risk_assessment", cascade="all, delete-orphan")


class RiskFactorModel(Base):
    """TABLE 10: risk_factors (Agent 5 Detailed Risk Factors)"""
    __tablename__ = "risk_factors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    risk_assessment_id = Column(Integer, ForeignKey("risk_assessments.id", ondelete="CASCADE"), nullable=False)
    category = Column(String(100), nullable=False)  # IDENTITY_RISK, INCOME_RISK, EMPLOYMENT_RISK, etc.
    severity = Column(String(50), default="LOW", nullable=False)  # LOW, MEDIUM, HIGH, CRITICAL
    points = Column(Float, default=0.0, nullable=False)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String(100), default="Agent 5 Risk Engine", nullable=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    finding_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    risk_assessment = relationship("RiskAssessmentModel", back_populates="risk_factors")


class FinalReportModel(Base):
    """TABLE 11: final_reports (Agent 6 Output)"""
    __tablename__ = "final_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    decision = Column(String(50), nullable=False)  # PASS, HUMAN_REVIEW, INSUFFICIENT_EVIDENCE
    decision_reason = Column(Text, nullable=True)
    risk_score = Column(Float, default=0.0, nullable=False)
    risk_level = Column(String(50), default="LOW", nullable=False)
    verification_coverage = Column(Float, default=0.0, nullable=False)
    consistency_score = Column(Float, nullable=True)
    document_summary = Column(Text, nullable=True)  # JSON encoded summary
    validation_summary = Column(Text, nullable=True)  # JSON encoded summary
    cross_document_summary = Column(Text, nullable=True)  # JSON encoded summary
    risk_summary = Column(Text, nullable=True)  # JSON encoded summary
    key_findings = Column(Text, nullable=True)  # JSON encoded list
    recommendations = Column(Text, nullable=True)  # JSON encoded list
    executive_summary = Column(Text, nullable=True)
    review_required = Column(Boolean, default=False, nullable=False)
    generated_by = Column(String(100), default="Agent 6 Final Report Decision Engine", nullable=True)
    generation_status = Column(String(50), default="SUCCESS", nullable=False)
    processing_time_ms = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="final_reports")


class ProcessingRunModel(Base):
    """TABLE 12: processing_runs (Execution Audit Trail)"""
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(50), default="STARTED", nullable=False)  # STARTED, PROCESSING, COMPLETED, FAILED
    current_agent = Column(String(50), default="agent_1", nullable=False)
    error_message = Column(Text, nullable=True)
    total_processing_time_ms = Column(Float, default=0.0, nullable=False)

    application = relationship("LoanApplication", back_populates="processing_runs")


# =============================================================================
# 13. LOAN POLICIES & RULES
# =============================================================================

class LoanPolicyModel(Base):
    """TABLE 13: loan_policies"""
    __tablename__ = "loan_policies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(100), unique=True, index=True, nullable=False)
    loan_type = Column(String(100), index=True, nullable=False)
    policy_name = Column(String(200), nullable=False)
    version = Column(String(50), default="1.0", nullable=False)
    description = Column(Text, nullable=True)
    effective_date = Column(String(50), nullable=True)
    source_type = Column(String(50), default="DEMO_POLICY", nullable=False)
    source_document = Column(String(200), nullable=True)
    source_section = Column(String(100), nullable=True)
    source_page = Column(Integer, nullable=True)
    status = Column(String(50), default="ACTIVE", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    rules = relationship("PolicyRuleModel", back_populates="policy", cascade="all, delete-orphan")
    required_documents = relationship("PolicyRequiredDocumentModel", back_populates="policy", cascade="all, delete-orphan")
    required_fields = relationship("PolicyRequiredFieldModel", back_populates="policy", cascade="all, delete-orphan")


class PolicyRuleModel(Base):
    """TABLE 14: policy_rules"""
    __tablename__ = "policy_rules"

    id = Column(Integer, primary_key=True, autoincrement=True)
    rule_id = Column(String(100), unique=True, index=True, nullable=False)
    policy_id = Column(String(100), ForeignKey("loan_policies.policy_id", ondelete="CASCADE"), nullable=False)
    rule_code = Column(String(100), index=True, nullable=False)
    category = Column(String(100), nullable=False)  # KYC, INCOME, AGE, PROPERTY, COLLATERAL, etc.
    field_name = Column(String(100), nullable=True)
    operator = Column(String(50), nullable=False)  # GTE, LTE, EQ, IN, EXISTS, MATCHES
    expected_value = Column(String(200), nullable=True)
    threshold_value = Column(Float, nullable=True)
    severity = Column(String(50), default="CRITICAL", nullable=False)  # CRITICAL, HIGH, MEDIUM, LOW
    mandatory = Column(Boolean, default=True, nullable=False)
    error_message = Column(Text, nullable=True)
    source_type = Column(String(50), default="DEMO_POLICY", nullable=False)
    source_document = Column(String(200), nullable=True)
    source_section = Column(String(100), nullable=True)
    source_page = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    policy = relationship("LoanPolicyModel", back_populates="rules")


class PolicyRequiredDocumentModel(Base):
    """TABLE 15: policy_required_documents"""
    __tablename__ = "policy_required_documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(100), ForeignKey("loan_policies.policy_id", ondelete="CASCADE"), nullable=False)
    loan_type = Column(String(100), nullable=False)
    slot_id = Column(String(100), nullable=False)
    document_type = Column(String(100), nullable=False)
    display_name = Column(String(200), nullable=False)
    required = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    policy = relationship("LoanPolicyModel", back_populates="required_documents")


class PolicyRequiredFieldModel(Base):
    """TABLE 16: policy_required_fields"""
    __tablename__ = "policy_required_fields"

    id = Column(Integer, primary_key=True, autoincrement=True)
    policy_id = Column(String(100), ForeignKey("loan_policies.policy_id", ondelete="CASCADE"), nullable=False)
    loan_type = Column(String(100), nullable=False)
    document_type = Column(String(100), nullable=False)
    field_name = Column(String(100), nullable=False)
    required = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    policy = relationship("LoanPolicyModel", back_populates="required_fields")


# =============================================================================
# 17. FIELD EVIDENCE & CITATIONS
# =============================================================================

class FieldEvidenceModel(Base):
    """TABLE 17: field_evidence"""
    __tablename__ = "field_evidence"

    id = Column(Integer, primary_key=True, autoincrement=True)
    evidence_id = Column(String(100), unique=True, index=True, nullable=False)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    slot_id = Column(String(100), nullable=True)
    document_type = Column(String(100), nullable=True)
    field_name = Column(String(100), nullable=False)
    extracted_value = Column(Text, nullable=True)  # Masked if sensitive
    raw_value = Column(Text, nullable=True)        # Masked if sensitive
    normalized_value = Column(Text, nullable=True)
    source_page = Column(Integer, default=1, nullable=False)
    snippet = Column(Text, nullable=True)
    bounding_box_json = Column(Text, nullable=True)
    extraction_method = Column(String(100), default="Regex / Key-Value Extractor", nullable=False)
    ocr_used = Column(Boolean, default=False, nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)
    validation_status = Column(String(50), default="UNVALIDATED", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="field_evidences")


# =============================================================================
# 18. DECISION GRAPHS
# =============================================================================

class DecisionGraphModel(Base):
    """TABLE 18: decision_graphs"""
    __tablename__ = "decision_graphs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_id = Column(String(100), unique=True, index=True, nullable=False)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    total_nodes = Column(Integer, default=0, nullable=False)
    total_edges = Column(Integer, default=0, nullable=False)
    passed_nodes = Column(Integer, default=0, nullable=False)
    failed_nodes = Column(Integer, default=0, nullable=False)
    warning_nodes = Column(Integer, default=0, nullable=False)
    info_nodes = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="decision_graphs")
    nodes = relationship("DecisionGraphNodeModel", back_populates="graph", cascade="all, delete-orphan")
    edges = relationship("DecisionGraphEdgeModel", back_populates="graph", cascade="all, delete-orphan")


class DecisionGraphNodeModel(Base):
    """TABLE 19: decision_graph_nodes"""
    __tablename__ = "decision_graph_nodes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_id = Column(String(100), ForeignKey("decision_graphs.graph_id", ondelete="CASCADE"), nullable=False)
    node_id = Column(String(100), nullable=False)
    stage_name = Column(String(100), nullable=False)
    label = Column(String(200), nullable=False)
    status = Column(String(50), nullable=False)  # PASS, FAIL, WARNING, INFO
    agent_name = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    evidence_count = Column(Integer, default=0, nullable=False)
    citation_count = Column(Integer, default=0, nullable=False)
    node_metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    graph = relationship("DecisionGraphModel", back_populates="nodes")


class DecisionGraphEdgeModel(Base):
    """TABLE 20: decision_graph_edges"""
    __tablename__ = "decision_graph_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    graph_id = Column(String(100), ForeignKey("decision_graphs.graph_id", ondelete="CASCADE"), nullable=False)
    source_node_id = Column(String(100), nullable=False)
    target_node_id = Column(String(100), nullable=False)
    edge_label = Column(String(100), nullable=True)
    edge_type = Column(String(50), default="DIRECTED", nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    graph = relationship("DecisionGraphModel", back_populates="edges")


# =============================================================================
# 21. ELIGIBILITY RESULTS & RULE RESULTS
# =============================================================================

class EligibilityResultModel(Base):
    """TABLE 21: eligibility_results"""
    __tablename__ = "eligibility_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eligibility_id = Column(String(100), unique=True, index=True, nullable=False)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    loan_type = Column(String(100), nullable=False)
    decision = Column(String(50), nullable=False)  # ELIGIBLE, NOT_ELIGIBLE, HUMAN_REVIEW_REQUIRED, INSUFFICIENT_EVIDENCE
    confidence = Column(Float, default=1.0, nullable=False)
    reasons_json = Column(Text, nullable=True)  # JSON array of reasons
    rules_evaluated_count = Column(Integer, default=0, nullable=False)
    rules_passed_count = Column(Integer, default=0, nullable=False)
    rules_failed_count = Column(Integer, default=0, nullable=False)
    rules_skipped_count = Column(Integer, default=0, nullable=False)
    processing_time_ms = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="eligibility_results")
    rule_results = relationship("EligibilityRuleResultModel", back_populates="eligibility_result", cascade="all, delete-orphan")


class EligibilityRuleResultModel(Base):
    """TABLE 22: eligibility_rule_results"""
    __tablename__ = "eligibility_rule_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    eligibility_id = Column(String(100), ForeignKey("eligibility_results.eligibility_id", ondelete="CASCADE"), nullable=False)
    rule_code = Column(String(100), nullable=False)
    category = Column(String(100), nullable=False)
    field_name = Column(String(100), nullable=True)
    operator = Column(String(50), nullable=False)
    expected_value = Column(String(200), nullable=True)
    actual_value = Column(String(200), nullable=True)
    status = Column(String(50), nullable=False)  # PASS, FAIL, SKIPPED, INSUFFICIENT_EVIDENCE
    mandatory = Column(Boolean, default=True, nullable=False)
    severity = Column(String(50), default="CRITICAL", nullable=False)
    failure_reason = Column(Text, nullable=True)
    evidence_id = Column(String(100), nullable=True)
    policy_rule_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    eligibility_result = relationship("EligibilityResultModel", back_populates="rule_results")


# =============================================================================
# 23. AGENT EXECUTION LOGS & TELEMETRY
# =============================================================================

class AgentExecutionLogModel(Base):
    """TABLE 23: agent_execution_logs"""
    __tablename__ = "agent_execution_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    agent_name = Column(String(100), nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime, nullable=True)
    duration_ms = Column(Float, default=0.0, nullable=False)
    status = Column(String(50), default="SUCCESS", nullable=False)
    input_summary_json = Column(Text, nullable=True)
    output_summary_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="agent_execution_logs")


class TelemetryMetricsModel(Base):
    """TABLE 24: telemetry_metrics"""
    __tablename__ = "telemetry_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    application_id = Column(String(100), ForeignKey("loan_applications.application_id", ondelete="CASCADE"), nullable=False)
    total_pipeline_duration_ms = Column(Float, default=0.0, nullable=False)
    agent_durations_json = Column(Text, nullable=True)  # JSON dict of agent durations
    ocr_duration_ms = Column(Float, default=0.0, nullable=False)
    validation_duration_ms = Column(Float, default=0.0, nullable=False)
    cross_doc_duration_ms = Column(Float, default=0.0, nullable=False)
    risk_duration_ms = Column(Float, default=0.0, nullable=False)
    eligibility_duration_ms = Column(Float, default=0.0, nullable=False)
    decision_graph_duration_ms = Column(Float, default=0.0, nullable=False)
    report_duration_ms = Column(Float, default=0.0, nullable=False)
    documents_count = Column(Integer, default=0, nullable=False)
    fields_count = Column(Integer, default=0, nullable=False)
    citations_count = Column(Integer, default=0, nullable=False)
    memory_mb = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    application = relationship("LoanApplication", back_populates="telemetry_metrics")


class User(Base):
    """TABLE 25: users (Bank Employees and Bank Managers)"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    full_name = Column(String(200), nullable=False)
    email = Column(String(200), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False)  # BANK_EMPLOYEE or BANK_MANAGER
    branch_id = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

