"""
===============================================================================
AGENT 5 — RISK & ANOMALY DETECTION ENGINE
===============================================================================
Single Python Implementation File for Agent 5.

Core Responsibilities:
1. Risk Detection (Document Risks, Validation Risks, Mismatch Risks, Domain Risks)
2. Anomaly Detection (Deterministic Identifier Checks, Structural & Semantic Anomalies)
3. Risk Deduplication (Prevents double-counting the same underlying problem between Agent 3 & Agent 4)
4. Deterministic Risk Scoring (0.0 to 100.0, capped at 100)
5. Risk Level Assignment (LOW: 0-24, MEDIUM: 25-49, HIGH: 50-74, CRITICAL: 75-100)
6. Domain-Specific Loan Rules (Agriculture, Vehicle, Gold, FD, Business, Education, Property, Home, Personal, Consumer Durable)
7. LangChain + Ollama Semantic Reasoning (with deterministic fallback when Ollama is offline)
8. Sensitive Data Masking (PAN, Aadhaar, Bank Account, FD, Chassis, Engine)
9. Downstream Contract: next_agent = "final_report_agent"
"""

import os
import sys
import re
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dotenv import load_dotenv

# Pydantic & LangGraph Imports
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Import project root and shared modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.state import (
    RiskFactor,
    RiskAssessmentResult,
    CrossDocumentResult,
    DocumentValidationResult,
    DocumentExtractionResult,
    DocumentClassificationResult,
    LoanDocumentState
)
from shared.policy import (
    get_loan_type_policy,
    LOAN_TYPE_NAMES,
    is_document_acceptable_for_requirement
)

# Optional LangChain Ollama import
try:
    from langchain_ollama import ChatOllama
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Agent5_RiskAnomalyAgent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

RISK_LEVEL_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "LOW": (0.0, 24.99),
    "MEDIUM": (25.0, 49.99),
    "HIGH": (50.0, 74.99),
    "CRITICAL": (75.0, 100.0)
}

RISK_POINTS: Dict[str, float] = {
    "CRITICAL_MISMATCH": 40.0,
    "HIGH_MISMATCH": 25.0,
    "MEDIUM_MISMATCH": 15.0,
    "LOW_MISMATCH": 5.0,
    "CRITICAL_VALIDATION_FAIL": 30.0,
    "HIGH_VALIDATION_FAIL": 20.0,
    "MEDIUM_VALIDATION_FAIL": 10.0,
    "LOW_VALIDATION_WARN": 5.0,
    "MISSING_MANDATORY_DOC": 25.0,
    "WRONG_MANDATORY_DOC": 25.0,
    "UNREADABLE_MANDATORY_DOC": 25.0,
    "MISSING_MANDATORY_FIELD": 10.0,
    "VERY_LOW_CONFIDENCE": 10.0,
}

SUPPORTED_RISK_CATEGORIES = {
    "IDENTITY_RISK",
    "INCOME_RISK",
    "EMPLOYMENT_RISK",
    "BANKING_RISK",
    "DOCUMENT_RISK",
    "CROSS_DOCUMENT_RISK",
    "PROPERTY_RISK",
    "VEHICLE_RISK",
    "EDUCATION_RISK",
    "BUSINESS_RISK",
    "GOLD_RISK",
    "AGRICULTURE_RISK",
    "FD_RISK",
    "FINANCIAL_RISK",
    "DATA_QUALITY_RISK",
    "OTHER"
}


# =============================================================================
# HELPER FUNCTIONS & PII MASKING
# =============================================================================

SENSITIVE_FIELD_PATTERNS = [
    r"pan", r"aadhaar", r"account", r"bank_account", r"fd_number", r"chassis", r"engine", r"ssn"
]


def get_obj_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Safely retrieves an attribute or key from either a Dict or a Pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if hasattr(obj, "model_dump"):
        return obj.model_dump().get(key, default)
    return default


def mask_sensitive_value(val: Any, field_name: str = "") -> str:
    """
    Masks sensitive values (PAN, Aadhaar, Bank Account, FD, Chassis, Engine)
    leaving only last 4 characters visible (e.g. XXXXXX7890).
    """
    if val is None:
        return "null"
    
    val_str = str(val).strip()
    if not val_str:
        return ""
    
    fn_lower = field_name.lower().strip()
    is_sensitive = any(p in fn_lower for p in SENSITIVE_FIELD_PATTERNS)
    
    if re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]$", val_str, re.IGNORECASE):
        is_sensitive = True
    if re.match(r"^[0-9]{12}$", val_str.replace(" ", "")):
        is_sensitive = True

    if is_sensitive and len(val_str) > 4:
        masked_prefix = "X" * (len(val_str) - 4)
        return f"{masked_prefix}{val_str[-4:]}"
    
    return val_str


def is_valid_value(val: Any) -> bool:
    """
    Strict Null Handling:
    Integer/float 0/0.0 and Boolean False ARE valid values.
    Returns False ONLY for None, empty strings, 'null', 'none', 'n/a', 'unknown'.
    """
    if val is None:
        return False
    if isinstance(val, (int, float, bool)):
        return True
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if not cleaned or cleaned in ["null", "none", "n/a", "na", "undefined", "unknown", "not available"]:
            return False
        return True
    if isinstance(val, (list, dict)):
        return len(val) > 0
    return True


# =============================================================================
# RISK & ANOMALY DETECTION ENGINE AGENT
# =============================================================================

class RiskAnomalyDetectionAgent:
    """
    Agent 5 — Risk & Anomaly Detection Engine.
    Executes deterministic risk aggregation, loan-specific domain policy evaluation,
    risk factor deduplication, and LangChain + Ollama semantic reasoning.
    """

    def __init__(self, ollama_url: Optional[str] = None, ollama_model: Optional[str] = None):
        self.ollama_url = ollama_url or OLLAMA_BASE_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Constructs the LangGraph workflow pipeline for Agent 5."""
        workflow = StateGraph(LoanDocumentState)

        workflow.add_node("receive_inputs", self._node_receive_inputs)
        workflow.add_node("collect_document_risks", self._node_collect_document_risks)
        workflow.add_node("collect_validation_risks", self._node_collect_validation_risks)
        workflow.add_node("collect_cross_document_risks", self._node_collect_cross_document_risks)
        workflow.add_node("collect_loan_specific_risks", self._node_collect_loan_specific_risks)
        workflow.add_node("deduplicate_and_score_risks", self._node_deduplicate_and_score_risks)
        workflow.add_node("semantic_anomaly_analysis", self._node_semantic_anomaly_analysis)
        workflow.add_node("prepare_agent6_output", self._node_prepare_agent6_output)

        workflow.add_edge(START, "receive_inputs")
        workflow.add_edge("receive_inputs", "collect_document_risks")
        workflow.add_edge("collect_document_risks", "collect_validation_risks")
        workflow.add_edge("collect_validation_risks", "collect_cross_document_risks")
        workflow.add_edge("collect_cross_document_risks", "collect_loan_specific_risks")
        workflow.add_edge("collect_loan_specific_risks", "deduplicate_and_score_risks")
        workflow.add_edge("deduplicate_and_score_risks", "semantic_anomaly_analysis")
        workflow.add_edge("semantic_anomaly_analysis", "prepare_agent6_output")
        workflow.add_edge("prepare_agent6_output", END)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # LANGGRAPH NODES
    # -------------------------------------------------------------------------

    def _node_receive_inputs(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 1: Receives and validates pipeline inputs."""
        app_id = state.get("application_id", f"APP-{int(time.time() * 1000)}")
        loan_type = state.get("loan_type", "personal_loan")
        logger.info(f"[Agent 5] Executing Risk & Anomaly Detection for {app_id} (Loan Type: {loan_type})")
        return {
            "application_id": app_id,
            "loan_type": loan_type,
            "processing_metrics": {"start_time": time.time()}
        }

    def _node_collect_document_risks(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 2: Evaluates document completeness and slot policy risks.
        Checks for missing required documents, wrong document uploads, or corrupt files.
        """
        loan_type = state.get("loan_type", "personal_loan")
        policy = get_loan_type_policy(loan_type)
        required_reqs = policy.get("required", [])

        classification_results = state.get("classification_results") or []
        has_class_input = state.get("_has_classification_input", False)
        
        doc_risks: List[RiskFactor] = []
        if not has_class_input:
            return {"doc_risks": doc_risks}

        rf_idx = 1
        for req in required_reqs:
            accepted = [a.lower().strip() for a in req.accepted_document_types]
            matched_c = next((c for c in classification_results if (get_obj_attr(c, "document_type") or "").lower() in accepted), None)
            
            if not matched_c:
                matched_c = next(
                    (c for c in classification_results if is_document_acceptable_for_requirement(get_obj_attr(c, "document_type"), req)),
                    None
                )

            if not matched_c:
                rf = RiskFactor(
                    factor_id=f"RF-DOC-{rf_idx:03d}",
                    category="DOCUMENT_RISK",
                    severity="HIGH",
                    title=f"Missing Mandatory Document: {req.display_name}",
                    description=f"The mandatory document '{req.display_name}' required for {LOAN_TYPE_NAMES.get(loan_type, loan_type)} was not provided.",
                    source="Agent 1 Document Policy",
                    field="document_type",
                    value_a=None,
                    value_b=req.display_name,
                    evidence=f"Required slot '{req.requirement_id}' is empty.",
                    points=RISK_POINTS["MISSING_MANDATORY_DOC"],
                    confidence=1.0,
                    recommendation=f"Upload a valid {req.display_name} to fulfill loan requirement."
                )
                doc_risks.append(rf)
                rf_idx += 1
            else:
                conf = get_obj_attr(matched_c, "confidence", 1.0)
                quality = get_obj_attr(matched_c, "content_quality", "good")
                filename = get_obj_attr(matched_c, "filename", "document")
                if conf < 0.5 or quality in ["corrupted", "empty"]:
                    rf = RiskFactor(
                        factor_id=f"RF-DOC-{rf_idx:03d}",
                        category="DOCUMENT_RISK",
                        severity="MEDIUM",
                        title=f"Low Document Quality: {filename}",
                        description=f"The uploaded document '{filename}' has low classification confidence ({conf*100:.1f}%) or poor content quality ({quality}).",
                        source="Agent 1 Document Classifier",
                        document_a=filename,
                        field="content_quality",
                        value_a=quality,
                        value_b="good",
                        evidence=f"Confidence: {conf}, Quality: {quality}",
                        points=RISK_POINTS["VERY_LOW_CONFIDENCE"],
                        confidence=0.9,
                        recommendation="Re-upload a clearer, higher-resolution copy of the document."
                    )
                    doc_risks.append(rf)
                    rf_idx += 1

        return {"doc_risks": doc_risks}

    def _node_collect_validation_risks(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 3: Evaluates Agent 3 Validation findings (FAIL or WARNING findings).
        Distinguishes DATA_QUALITY_RISK (missing field / extraction issue) from actual validation errors.
        """
        validation_results = state.get("validation_results") or []
        val_risks: List[RiskFactor] = []
        rf_idx = 1

        for v_res in validation_results:
            doc_type = get_obj_attr(v_res, "document_type", "document")
            filename = get_obj_attr(v_res, "filename", "")
            findings = get_obj_attr(v_res, "findings") or []

            for f in findings:
                status = get_obj_attr(f, "status")
                if status not in ["FAIL", "WARNING"]:
                    continue

                field_name = get_obj_attr(f, "field_name", "unknown_field")
                severity = str(get_obj_attr(f, "severity", "MEDIUM")).upper()
                msg = get_obj_attr(f, "message", "Validation rule failure")
                val_val = get_obj_attr(f, "value")
                finding_type = get_obj_attr(f, "finding_type", "OTHER_VALIDATION_ERROR")

                is_extraction_issue = (
                    finding_type in ["MISSING_REQUIRED_FIELD", "FORMAT_ERROR", "READABILITY_ERROR"] or
                    "missing or null" in msg.lower() or
                    "could not be extracted" in msg.lower()
                )

                if is_extraction_issue:
                    cat = "DATA_QUALITY_RISK"
                    risk_sev = "MEDIUM" if status == "FAIL" else "LOW"
                    points = 10.0 if status == "FAIL" else 5.0
                    title = f"Data Quality Issue: {field_name.replace('_', ' ').title()} in {filename or doc_type}"
                    desc = f"Required field '{field_name}' could not be reliably extracted from {filename or doc_type}."
                    rec = f"Review document extraction for '{field_name}' in {filename or doc_type}."
                else:
                    cat = self._determine_category_for_field(field_name, doc_type)
                    if status == "FAIL":
                        points = RISK_POINTS.get(f"{severity}_VALIDATION_FAIL", 20.0)
                        risk_sev = "HIGH" if severity in ["HIGH", "CRITICAL"] else "MEDIUM"
                    else:  # WARNING
                        points = RISK_POINTS.get("LOW_VALIDATION_WARN", 5.0)
                        risk_sev = "LOW"
                    title = f"Validation Failure: {field_name.replace('_', ' ').title()} in {filename or doc_type}"
                    desc = msg
                    rec = f"Review extracted field '{field_name}' in {filename or doc_type} for correctness."

                rf = RiskFactor(
                    factor_id=f"RF-VAL-{rf_idx:03d}",
                    category=cat,
                    severity=risk_sev,
                    title=title,
                    description=desc,
                    source="Agent 3 — Validation",
                    document_a=filename or doc_type,
                    field=field_name,
                    value_a=mask_sensitive_value(val_val, field_name),
                    evidence=f"Field '{field_name}' validation finding ({finding_type}).",
                    points=points,
                    confidence=0.95,
                    recommendation=rec
                )
                val_risks.append(rf)
                rf_idx += 1

        return {"val_risks": val_risks}

    def _node_collect_cross_document_risks(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 4: Evaluates Agent 4 Cross-Document Mismatch findings.
        CRITICAL PRINCIPLE: UNVERIFIABLE is NOT mismatch, NOT_APPLICABLE is NOT risk.
        """
        cross_doc_res = state.get("cross_document_results") or {}
        findings = get_obj_attr(cross_doc_res, "findings") or []
        
        cross_risks: List[RiskFactor] = []
        rf_idx = 1

        for f in findings:
            comp_res = get_obj_attr(f, "comparison_result", "")
            if comp_res != "MISMATCH":
                continue

            field_name = get_obj_attr(f, "field_name", "unknown_field")
            doc1 = get_obj_attr(f, "doc1_type", "")
            val1 = get_obj_attr(f, "doc1_value")
            doc2 = get_obj_attr(f, "doc2_type", "")
            val2 = get_obj_attr(f, "doc2_value")
            severity = str(get_obj_attr(f, "severity", "HIGH")).upper()
            msg = get_obj_attr(f, "message", "Cross-document data mismatch detected")

            points = RISK_POINTS.get(f"{severity}_MISMATCH", 25.0)
            cat = self._determine_category_for_field(field_name, f"{doc1}_{doc2}")

            if any(k in field_name.lower() for k in ["chassis", "engine", "fd_number", "gstin", "pan_number"]):
                severity = "CRITICAL"
                points = RISK_POINTS["CRITICAL_MISMATCH"]

            rf = RiskFactor(
                factor_id=f"RF-XDOC-{rf_idx:03d}",
                category=cat,
                severity=severity,
                title=f"Cross-Document Mismatch: {field_name.replace('_', ' ').title()}",
                description=msg,
                source="Agent 4 — Cross-Document Verification",
                document_a=doc1,
                document_b=doc2,
                field=field_name,
                value_a=mask_sensitive_value(val1, field_name),
                value_b=mask_sensitive_value(val2, field_name),
                evidence=f"Conflict between {doc1} ('{mask_sensitive_value(val1, field_name)}') and {doc2} ('{mask_sensitive_value(val2, field_name)}')",
                points=points,
                confidence=get_obj_attr(f, "confidence", 0.95),
                recommendation=f"Verify conflicting {field_name.replace('_', ' ')} between {doc1} and {doc2}."
            )
            cross_risks.append(rf)
            rf_idx += 1

        return {"cross_risks": cross_risks}

    def _node_collect_loan_specific_risks(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 5: Evaluates domain-specific loan policy rules across all 10 loan types.
        """
        loan_type = (state.get("loan_type") or "personal_loan").lower().strip()
        extraction_results = state.get("extraction_results") or []
        
        doc_map: Dict[str, Dict[str, Any]] = {}
        for ext in extraction_results:
            dt = (get_obj_attr(ext, "document_type") or "").lower()
            fields_dict = get_obj_attr(ext, "fields") or {}
            val_map = {}
            for k, v in fields_dict.items():
                v_val = get_obj_attr(v, "value") if (isinstance(v, dict) or hasattr(v, "value")) else v
                val_map[k] = v_val
            doc_map[dt] = val_map

        domain_risks: List[RiskFactor] = []
        rf_idx = 1

        if loan_type == "vehicle_loan":
            quot_chassis = doc_map.get("vehicle_quotation", {}).get("chassis_number")
            inv_chassis = doc_map.get("vehicle_invoice", {}).get("chassis_number")

            if is_valid_value(quot_chassis) and is_valid_value(inv_chassis):
                if str(quot_chassis).strip().upper() != str(inv_chassis).strip().upper():
                    rf = RiskFactor(
                        factor_id=f"RF-SPEC-{rf_idx:03d}",
                        category="VEHICLE_RISK",
                        severity="CRITICAL",
                        title="Vehicle Chassis Number Mismatch",
                        description=f"Vehicle Quotation chassis ({mask_sensitive_value(quot_chassis, 'chassis')}) conflicts with Invoice chassis ({mask_sensitive_value(inv_chassis, 'chassis')}).",
                        source="Agent 5 Vehicle Domain Rule",
                        document_a="vehicle_quotation",
                        document_b="vehicle_invoice",
                        field="chassis_number",
                        value_a=mask_sensitive_value(quot_chassis, "chassis"),
                        value_b=mask_sensitive_value(inv_chassis, "chassis"),
                        evidence="Strict vehicle identifier mismatch.",
                        points=RISK_POINTS["CRITICAL_MISMATCH"],
                        confidence=1.0,
                        recommendation="Reject or require official dealership clarification for chassis number discrepancy."
                    )
                    domain_risks.append(rf)
                    rf_idx += 1

        elif loan_type == "gold_loan":
            val_purity = doc_map.get("jewellery_valuation_report", {}).get("gold_purity")
            sec_purity = doc_map.get("gold_security_document", {}).get("gold_purity")
            
            if is_valid_value(val_purity) and is_valid_value(sec_purity):
                if str(val_purity).strip().upper() != str(sec_purity).strip().upper():
                    rf = RiskFactor(
                        factor_id=f"RF-SPEC-{rf_idx:03d}",
                        category="GOLD_RISK",
                        severity="HIGH",
                        title="Gold Purity Assessment Mismatch",
                        description=f"Valuation report purity ({val_purity}) does not match security pledge document purity ({sec_purity}).",
                        source="Agent 5 Gold Domain Rule",
                        document_a="jewellery_valuation_report",
                        document_b="gold_security_document",
                        field="gold_purity",
                        value_a=val_purity,
                        value_b=sec_purity,
                        evidence="Gold purity rating conflict.",
                        points=RISK_POINTS["HIGH_MISMATCH"],
                        confidence=1.0,
                        recommendation="Re-assess jewellery purity with certified appraiser."
                    )
                    domain_risks.append(rf)
                    rf_idx += 1

        elif loan_type == "loan_against_fd":
            fd1 = doc_map.get("fixed_deposit_certificate", {}).get("fd_number") or doc_map.get("fixed_deposit_receipt", {}).get("fd_number")
            fd2 = doc_map.get("fd_statement", {}).get("fd_number")

            if is_valid_value(fd1) and is_valid_value(fd2):
                if str(fd1).strip().upper() != str(fd2).strip().upper():
                    rf = RiskFactor(
                        factor_id=f"RF-SPEC-{rf_idx:03d}",
                        category="FD_RISK",
                        severity="CRITICAL",
                        title="Fixed Deposit Certificate Number Mismatch",
                        description=f"FD Certificate number ({mask_sensitive_value(fd1, 'fd_number')}) does not match FD Statement ({mask_sensitive_value(fd2, 'fd_number')}).",
                        source="Agent 5 FD Domain Rule",
                        document_a="fixed_deposit_certificate",
                        document_b="fd_statement",
                        field="fd_number",
                        value_a=mask_sensitive_value(fd1, "fd_number"),
                        value_b=mask_sensitive_value(fd2, "fd_number"),
                        evidence="FD instrument account number mismatch.",
                        points=RISK_POINTS["CRITICAL_MISMATCH"],
                        confidence=1.0,
                        recommendation="Verify underlying FD account ownership and receipt validity."
                    )
                    domain_risks.append(rf)
                    rf_idx += 1

        elif loan_type == "agriculture_loan":
            land_survey = doc_map.get("land_record", {}).get("survey_number")
            crop_survey = doc_map.get("cultivation_record", {}).get("survey_number")

            if is_valid_value(land_survey) and is_valid_value(crop_survey):
                clean_ls = re.sub(r"[\s\-\/]", "", str(land_survey).lower())
                clean_cs = re.sub(r"[\s\-\/]", "", str(crop_survey).lower())
                if clean_ls != clean_cs:
                    rf = RiskFactor(
                        factor_id=f"RF-SPEC-{rf_idx:03d}",
                        category="AGRICULTURE_RISK",
                        severity="HIGH",
                        title="Agricultural Survey Number Discrepancy",
                        description=f"Land Record survey number ({land_survey}) differs from Cultivation Record survey number ({crop_survey}).",
                        source="Agent 5 Agriculture Domain Rule",
                        document_a="land_record",
                        document_b="cultivation_record",
                        field="survey_number",
                        value_a=land_survey,
                        value_b=crop_survey,
                        evidence="Land plot survey identifier mismatch.",
                        points=RISK_POINTS["HIGH_MISMATCH"],
                        confidence=0.95,
                        recommendation="Verify village revenue records for plot subdivision / survey number."
                    )
                    domain_risks.append(rf)
                    rf_idx += 1

        elif loan_type == "business_loan":
            gst1 = doc_map.get("gst_certificate", {}).get("gstin")
            gst2 = doc_map.get("gst_return", {}).get("gstin") or doc_map.get("business_bank_statement", {}).get("gstin")

            if is_valid_value(gst1) and is_valid_value(gst2):
                if str(gst1).strip().upper() != str(gst2).strip().upper():
                    rf = RiskFactor(
                        factor_id=f"RF-SPEC-{rf_idx:03d}",
                        category="BUSINESS_RISK",
                        severity="CRITICAL",
                        title="Business GSTIN Identifier Mismatch",
                        description=f"GST Certificate GSTIN ({gst1}) differs from return/bank GSTIN ({gst2}).",
                        source="Agent 5 Business Domain Rule",
                        document_a="gst_certificate",
                        document_b="gst_return",
                        field="gstin",
                        value_a=gst1,
                        value_b=gst2,
                        evidence="Business Tax Identifier conflict.",
                        points=RISK_POINTS["CRITICAL_MISMATCH"],
                        confidence=1.0,
                        recommendation="Cross-check GST portal registration details."
                    )
                    domain_risks.append(rf)
                    rf_idx += 1

        return {"domain_risks": domain_risks}

    def _node_deduplicate_and_score_risks(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 6: Risk Deduplication & Deterministic Risk Scoring.
        Prevents double-counting the exact same problem reported by both Agent 3 and Agent 4.
        Caps risk_score at 100.0. Assigns risk_level (LOW, MEDIUM, HIGH, CRITICAL).
        """
        doc_risks: List[RiskFactor] = state.get("doc_risks") or []
        val_risks: List[RiskFactor] = state.get("val_risks") or []
        cross_risks: List[RiskFactor] = state.get("cross_risks") or []
        domain_risks: List[RiskFactor] = state.get("domain_risks") or []

        all_raw_risks = doc_risks + val_risks + cross_risks + domain_risks

        dedup_map: Dict[Tuple[str, str, str], RiskFactor] = {}

        for rf in all_raw_risks:
            cat = get_obj_attr(rf, "category", "OTHER").upper()
            field_key = (get_obj_attr(rf, "field") or "general").lower().strip()
            title_norm = re.sub(r"[^a-z0-9]", "", (get_obj_attr(rf, "title") or "").lower())[:30]
            if field_key and field_key != "general":
                key = (cat, field_key, "")
            else:
                key = (cat, field_key, title_norm)

            if key not in dedup_map:
                dedup_map[key] = rf
            else:
                existing = dedup_map[key]
                if get_obj_attr(rf, "points", 0.0) > get_obj_attr(existing, "points", 0.0):
                    dedup_map[key] = rf
                else:
                    existing_src = get_obj_attr(existing, "source", "")
                    rf_src = get_obj_attr(rf, "source", "")
                    if rf_src and rf_src not in existing_src:
                        if isinstance(existing, dict):
                            existing["source"] = f"{existing_src}, {rf_src}"
                        else:
                            existing.source = f"{existing_src}, {rf_src}"

        dedup_risks = list(dedup_map.values())

        for idx, rf in enumerate(dedup_risks, start=1):
            if isinstance(rf, dict):
                rf["factor_id"] = f"RF-{idx:03d}"
            else:
                rf.factor_id = f"RF-{idx:03d}"

        total_points = sum(get_obj_attr(rf, "points", 0.0) for rf in dedup_risks)
        risk_score = min(total_points, 100.0)

        crit_count = sum(1 for rf in dedup_risks if get_obj_attr(rf, "severity") == "CRITICAL")
        high_count = sum(1 for rf in dedup_risks if get_obj_attr(rf, "severity") == "HIGH")
        med_count = sum(1 for rf in dedup_risks if get_obj_attr(rf, "severity") == "MEDIUM")
        low_count = sum(1 for rf in dedup_risks if get_obj_attr(rf, "severity") == "LOW")

        risk_level = "LOW"
        for level, (min_p, max_p) in RISK_LEVEL_THRESHOLDS.items():
            if min_p <= risk_score <= max_p:
                risk_level = level
                break

        cross_doc_res = state.get("cross_document_results") or {}
        coverage = get_obj_attr(cross_doc_res, "verification_coverage", 0.0)
        consistency = get_obj_attr(cross_doc_res, "consistency_score")

        if risk_level in ["HIGH", "CRITICAL"]:
            rec_action = "HUMAN_REVIEW_REQUIRED"
        elif risk_level == "MEDIUM":
            rec_action = "REVIEW_RECOMMENDED"
        else:
            if coverage < 80.0 and len(dedup_risks) > 0:
                rec_action = "REVIEW_RECOMMENDED"
            else:
                rec_action = "PROCEED_TO_REPORT"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "total_risk_factors": len(dedup_risks),
            "critical_count": crit_count,
            "high_count": high_count,
            "medium_count": med_count,
            "low_count": low_count,
            "risk_factors": dedup_risks,
            "recommended_action": rec_action,
            "verification_coverage": coverage,
            "consistency_score": consistency
        }

    def _node_semantic_anomaly_analysis(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 7: LangChain + Ollama Semantic Reasoning.
        Generates executive risk summary and evaluates complex semantic anomalies.
        Includes robust exception handling for when Ollama is offline/unavailable.
        """
        app_id = state.get("application_id", "")
        loan_type = state.get("loan_type", "")
        risk_score = state.get("risk_score", 0.0)
        risk_level = state.get("risk_level", "LOW")
        risk_factors = state.get("risk_factors") or []
        rec_action = state.get("recommended_action", "PROCEED_TO_REPORT")
        coverage = state.get("verification_coverage", 0.0)
        consistency = state.get("consistency_score")

        anomalies_list = []
        rf_dumps = []
        for rf in risk_factors:
            f_id = get_obj_attr(rf, "factor_id")
            title = get_obj_attr(rf, "title")
            cat = get_obj_attr(rf, "category")
            sev = get_obj_attr(rf, "severity")
            desc = get_obj_attr(rf, "description")
            anomalies_list.append({
                "factor_id": f_id,
                "title": title,
                "category": cat,
                "severity": sev,
                "description": desc
            })
            rf_dumps.append(rf.model_dump() if isinstance(rf, BaseModel) else rf)

        summary_text = ""
        reasoning_status = "UNAVAILABLE"

        if LANGCHAIN_AVAILABLE and self.ollama_url:
            try:
                llm = ChatOllama(
                    base_url=self.ollama_url,
                    model=self.ollama_model,
                    temperature=0.0
                )

                prompt_content = f"""
Application ID: {app_id}
Loan Type: {loan_type}
Risk Score: {risk_score}/100 ({risk_level})
Verification Coverage: {coverage:.1f}%
Consistency Score: {consistency if consistency is not None else 'N/A'}
Total Risk Factors: {len(risk_factors)}
Recommended Action: {rec_action}

Active Risk Factors:
{json.dumps(rf_dumps, indent=2)}

Task: Provide a concise 3-4 sentence professional executive risk assessment summary for this loan application. Highlight the primary risk drivers or confirm if the application exhibits low risk and acceptable document consistency. Do NOT use markdown headers or lists.
"""
                messages = [
                    SystemMessage(content="You are an expert financial risk officer conducting automated loan document anomaly evaluation."),
                    HumanMessage(content=prompt_content)
                ]
                
                response = llm.invoke(messages)
                if response and response.content:
                    summary_text = response.content.strip()
                    reasoning_status = "AVAILABLE"
            except Exception as e:
                logger.warning(f"[Agent 5] Ollama LLM invocation unavailable or failed ({e}). Using deterministic fallback.")
                reasoning_status = "UNAVAILABLE"

        if not summary_text:
            if risk_level == "LOW":
                summary_text = f"Application {app_id} for {LOAN_TYPE_NAMES.get(loan_type, loan_type)} demonstrates low overall risk (Score: {risk_score:.1f}/100) with high cross-document consistency. Verification coverage is {coverage:.1f}%."
            elif risk_level == "MEDIUM":
                summary_text = f"Application {app_id} exhibits moderate risk (Score: {risk_score:.1f}/100) with {len(risk_factors)} active risk factors detected. Review of document discrepancies is recommended."
            else:
                summary_text = f"Application {app_id} displays elevated {risk_level} risk (Score: {risk_score:.1f}/100). {len(risk_factors)} critical/high risk factors were detected. Manual underwriter review is required."

        return {
            "overall_summary": summary_text,
            "anomalies": anomalies_list,
            "semantic_reasoning_status": reasoning_status
        }

    def _node_prepare_agent6_output(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 8: Formats final RiskAssessmentResult output model.
        Prepares contract next_agent = "final_report_agent".
        """
        start_time = state.get("processing_metrics", {}).get("start_time", time.time())
        proc_time_ms = round((time.time() - start_time) * 1000, 2)

        rf_objs = []
        for rf in (state.get("risk_factors") or []):
            if isinstance(rf, RiskFactor):
                rf_objs.append(rf)
            elif isinstance(rf, dict):
                rf_objs.append(RiskFactor(**rf))

        risk_res = RiskAssessmentResult(
            application_id=state.get("application_id", ""),
            loan_type=state.get("loan_type", "personal_loan"),
            risk_score=state.get("risk_score", 0.0),
            risk_level=state.get("risk_level", "LOW"),
            total_risk_factors=state.get("total_risk_factors", 0),
            critical_count=state.get("critical_count", 0),
            high_count=state.get("high_count", 0),
            medium_count=state.get("medium_count", 0),
            low_count=state.get("low_count", 0),
            risk_factors=rf_objs,
            anomalies=state.get("anomalies", []),
            overall_summary=state.get("overall_summary", ""),
            recommended_action=state.get("recommended_action", "PROCEED_TO_REPORT"),
            verification_coverage=state.get("verification_coverage", 0.0),
            consistency_score=state.get("consistency_score"),
            semantic_reasoning_status=state.get("semantic_reasoning_status", "UNAVAILABLE"),
            processing_time_ms=proc_time_ms,
            errors=state.get("errors", []),
            next_agent="final_report_agent"
        )

        logger.info(f"[Agent 5] Complete. Risk Score: {risk_res.risk_score}/100 ({risk_res.risk_level}), Factors: {risk_res.total_risk_factors}, Action: {risk_res.recommended_action}")
        return {"risk_assessment_results": risk_res.model_dump()}

    # -------------------------------------------------------------------------
    # PUBLIC API METHOD
    # -------------------------------------------------------------------------

    def process(
        self,
        cross_document_results: Union[Dict[str, Any], CrossDocumentResult],
        validation_results: List[Union[Dict[str, Any], DocumentValidationResult]] = None,
        extraction_results: List[Union[Dict[str, Any], DocumentExtractionResult]] = None,
        classification_results: List[Union[Dict[str, Any], DocumentClassificationResult]] = None,
        loan_type: str = "personal_loan",
        application_id: Optional[str] = None
    ) -> RiskAssessmentResult:
        app_id = application_id or get_obj_attr(cross_document_results, "application_id") or f"APP-{int(time.time() * 1000)}"
        l_type = loan_type or get_obj_attr(cross_document_results, "loan_type") or "personal_loan"
        cross_dict = cross_document_results.model_dump() if isinstance(cross_document_results, BaseModel) else (cross_document_results or {})
        val_dicts = [v.model_dump() if isinstance(v, BaseModel) else v for v in (validation_results or [])]
        ext_dicts = [e.model_dump() if isinstance(e, BaseModel) else e for e in (extraction_results or [])]
        class_dicts = [c.model_dump() if isinstance(c, BaseModel) else c for c in (classification_results or [])]
        has_class_input = classification_results is not None

        state: Dict[str, Any] = {
            "application_id": app_id,
            "loan_type": l_type,
            "classification_results": class_dicts,
            "extraction_results": ext_dicts,
            "validation_results": val_dicts,
            "cross_document_results": cross_dict,
            "errors": [],
            "processing_metrics": {},
            "next_agent": "risk_anomaly_agent",
            "_has_classification_input": has_class_input
        }

        # Execute node pipeline with complete state accumulation
        state.update(self._node_receive_inputs(state))
        state.update(self._node_collect_document_risks(state))
        state.update(self._node_collect_validation_risks(state))
        state.update(self._node_collect_cross_document_risks(state))
        state.update(self._node_collect_loan_specific_risks(state))
        state.update(self._node_deduplicate_and_score_risks(state))
        state.update(self._node_semantic_anomaly_analysis(state))
        state.update(self._node_prepare_agent6_output(state))

        risk_data = state.get("risk_assessment_results") or {}
        return RiskAssessmentResult(**risk_data)

    def _determine_category_for_field(self, field_name: str, context: str) -> str:
        """Maps a field name and context to a standard supported Risk Category."""
        fn = field_name.lower()
        ctx = context.lower()

        if any(k in fn for k in ["name", "pan", "aadhaar", "dob", "identity", "applicant"]):
            return "IDENTITY_RISK"
        if any(k in fn for k in ["income", "salary", "gross", "net", "pay", "ctc"]):
            return "INCOME_RISK"
        if any(k in fn for k in ["employer", "company", "designation", "employment"]):
            return "EMPLOYMENT_RISK"
        if any(k in fn for k in ["account", "bank", "balance", "credit", "ifsc"]):
            return "BANKING_RISK"
        if any(k in fn for k in ["property", "title", "deed", "address", "building"]):
            return "PROPERTY_RISK"
        if any(k in fn for k in ["chassis", "engine", "vehicle", "registration"]):
            return "VEHICLE_RISK"
        if any(k in fn for k in ["student", "admission", "co_applicant", "university", "academic"]):
            return "EDUCATION_RISK"
        if any(k in fn for k in ["gst", "business", "tax_return", "pnl", "balance_sheet"]):
            return "BUSINESS_RISK"
        if any(k in fn for k in ["gold", "purity", "weight", "jewellery"]):
            return "GOLD_RISK"
        if any(k in fn for k in ["survey", "crop", "cultivated", "land", "farmer"]):
            return "AGRICULTURE_RISK"
        if any(k in fn for k in ["fd", "deposit", "maturity"]):
            return "FD_RISK"
        if any(k in fn for k in ["document", "unreadable", "quality", "format"]):
            return "DOCUMENT_RISK"

        return "CROSS_DOCUMENT_RISK"
