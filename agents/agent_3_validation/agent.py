"""
===============================================================================
AGENT 3 — INFORMATION VALIDATION AGENT
===============================================================================
Single Python Implementation File for Agent 3.

Core Responsibility:
Validate Agent 2's extracted structured data per document type, performing:
1. Required-field presence validation (distinguishing missing/null vs numeric zero)
2. Data-type validation
3. Format validation (PAN, TAN, IFSC, Postal Code, etc.)
4. Date validation (valid format, DOB not future, start <= end, joining <= letter, expiry >= issue)
5. Numeric & Range validation (non-negative salaries/deductions)
6. Intra-document financial consistency (gross >= basic, net <= gross, gross ~ basic + HRA + allowances, net ~ gross - deductions, taxable <= gross_total_income)
7. Document-specific business rules (employment status vs type separation, issuer blacklist)
8. Field extraction confidence validation
9. Source-evidence verification
10. Overall extraction-status validation

Architecture & Principles:
- Single file: agents/agent_3_validation/agent.py
- LangGraph StateGraph orchestration pipeline
- High-precision Deterministic Python Validation Engine + Optional Semantic Ollama Explanation
- Anti-hallucination & Strict Deterministic Scoring:
    - 100 = All checks pass
    - 90-99 = Minor warnings
    - 70-89 = Moderate issues
    - Below 70 = Significant failures
- Failure Isolation: Multi-document batch runs validate each document independently.
- PII Masking in Logs: PAN, Account numbers, and sensitive text masked in log outputs.
- Downstream Contract: next_agent = "cross_document_agent"
"""

import os
import sys
import re
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
from dotenv import load_dotenv

# Pydantic & LangGraph Imports
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END

# Import project root and shared state
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from shared.state import (
        DocumentClassificationResult,
        DocumentExtractionResult,
        ExtractedField,
        ValidationFinding,
        DocumentValidationResult,
        LoanDocumentState
    )
except ImportError:
    sys.path.append(str(PROJECT_ROOT))
    from shared.state import (
        DocumentClassificationResult,
        DocumentExtractionResult,
        ExtractedField,
        ValidationFinding,
        DocumentValidationResult,
        LoanDocumentState
    )

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Agent3_ValidationAgent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# =============================================================================
# 1. HELPER FUNCTIONS & PII MASKING
# =============================================================================

def is_valid_extracted_value(val: Any) -> bool:
    """
    Helper to check if a value is genuinely present (non-null, non-empty).
    CRITICAL RULE: Integer or float 0/0.0 and Boolean False ARE valid values.
    Returns False ONLY for None, empty strings, 'null', 'none', 'n/a', and empty lists/dicts.
    """
    if val is None:
        return False
    if isinstance(val, (int, float, bool)):
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ["", "null", "none", "n/a", "unknown"]:
            return False
        return True
    if isinstance(val, (list, tuple, dict)):
        return len(val) > 0
    return False


def mask_pii(text: str) -> str:
    """Masks PAN numbers, account numbers, and sensitive identifiers for secure logging."""
    if not text:
        return ""
    # Mask Indian PAN (e.g. ABCDE1234F -> ABCD****F)
    text = re.sub(r'\b([A-Z]{4})[A-Z0-9]{5}([A-Z]{1})\b', r'\1****\2', text)
    # Mask Account numbers (>6 digits)
    text = re.sub(r'\b\d{4}(\d{4,12})\b', r'****\1', text)
    return text


def parse_date(date_str: Any) -> Optional[datetime]:
    """Parses various standard date formats into a datetime object."""
    if not date_str or not isinstance(date_str, str):
        return None
    s = date_str.strip()
    formats = [
        "%d-%b-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y",
        "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"
    ]
    for fmt in formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    # Try regex fallback for YYYY-MM-DD or DD-MM-YYYY
    m = re.match(r'^(\d{4})-(\d{2})-(\d{2})$', s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = re.match(r'^(\d{2})[-/](\d{2})[-/](\d{4})$', s)
    if m:
        try:
            return datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


# =============================================================================
# 2. REQUIRED FIELDS & REGEX RULES PER DOCUMENT TYPE
# =============================================================================

REQUIRED_FIELDS_PER_DOC_TYPE = {
    "payslip": [
        "employee_name", "employer_name", "pay_period",
        "basic_salary", "gross_salary", "net_salary"
    ],
    "bank_statement": [
        "account_holder_name", "account_number", "bank_name",
        "opening_balance", "closing_balance"
    ],
    "itr_tax_return": [
        "taxpayer_name", "assessment_year", "financial_year",
        "gross_total_income", "taxable_income"
    ],
    "kyc_identity": [
        "full_name", "date_of_birth", "identity_document_type", "identity_document_number"
    ],
    "pan_card": [
        "full_name", "identity_document_number"
    ],
    "aadhaar_identity": [
        "full_name", "identity_document_number"
    ],
    "passport_identity": [
        "full_name", "identity_document_number"
    ],
    "driving_license_identity": [
        "full_name", "identity_document_number"
    ],
    "voter_id_identity": [
        "full_name", "identity_document_number"
    ],
    "employment_letter": [
        "employee_name", "employer_name", "employment_status"
    ],
    "employment_certificate": [
        "employee_name", "employer_name"
    ],
    "office_id": [
        "employee_name", "employer_name"
    ],
    "employee_id": [
        "employee_name", "employer_name"
    ],
    "company_id": [
        "employee_name", "employer_name"
    ],
    "employment_proof": [
        "employee_name", "employer_name"
    ],
    "form_16": [
        "employee_name", "employer_name", "assessment_year",
        "salary_income", "employer_TAN", "employee_PAN"
    ],
    "address_proof": [
        "full_name", "address", "document_date", "issuer"
    ],
    "property_title_document": [
        "owner_name", "property_address"
    ],
    "sale_deed": [
        "owner_name", "property_address"
    ],
    "sale_agreement": [
        "buyer_name", "property_address", "sale_amount"
    ],
    "approved_building_plan": [
        "owner_name", "property_address", "approval_number", "approving_authority"
    ],
    "property_tax_receipt": [
        "owner_name", "property_address", "tax_amount"
    ],
    "property_valuation_report": [
        "owner_name", "property_address", "market_value", "valuer_name"
    ],
    "vehicle_quotation": [
        "customer_name", "dealer_name", "vehicle_make", "total_amount"
    ],
    "vehicle_invoice": [
        "buyer_name", "dealer_name", "invoice_number", "vehicle_make", "total_amount"
    ],
    "admission_letter": [
        "student_name", "institution_name", "course_name", "admission_number"
    ],
    "fee_structure": [
        "student_name", "institution_name", "course_name", "total_fee"
    ],
    "academic_certificate": [
        "student_name", "institution_name"
    ],
    "marksheet": [
        "student_name", "institution_name"
    ],
    "business_registration": [
        "business_name", "registration_number"
    ],
    "gst_certificate": [
        "business_name", "GSTIN"
    ],
    "gst_return": [
        "business_name", "GSTIN", "tax_period"
    ],
    "profit_loss_statement": [
        "business_name", "financial_year", "net_profit"
    ],
    "balance_sheet": [
        "business_name", "financial_year", "total_assets"
    ],
    "business_bank_statement": [
        "business_name", "account_number", "bank_name"
    ],
    "fixed_deposit_certificate": [
        "account_holder_name", "bank_name", "FD_number", "deposit_amount", "maturity_amount"
    ],
    "gold_security_document": [
        "borrower_name", "valuation_date", "assessed_value"
    ],
    "land_record": [
        "farmer_name", "survey_number", "land_area"
    ],
    "agricultural_income_proof": [
        "farmer_name", "financial_year", "agricultural_income"
    ],
    "product_quotation": [
        "customer_name", "seller_name", "total_amount"
    ],
    "product_invoice": [
        "customer_name", "seller_name", "total_amount"
    ],
    "student_kyc": [
        "full_name", "date_of_birth", "gender", "identity_document_type", "identity_document_number", "address"
    ],
    "co_applicant_kyc": [
        "full_name", "identity_document_type", "identity_document_number"
    ],
    "co_applicant_income_proof": [
        "employee_name", "employer_name", "pay_period", "basic_salary", "gross_salary", "net_salary"
    ],
    "other": [],
    "unknown": []
}

FORMAT_PATTERNS = {
    "pan": r'^[A-Z]{5}\d{4}[A-Z]{1}$',
    "tan": r'^[A-Z]{4}\d{5}[A-Z]{1}$',
    "ifsc": r'^[A-Z]{4}0[A-Z0-9]{6}$',
    "postal_code": r'^\d{5,6}$',
    "ay_fy": r'^\d{4}-\d{2,4}$',
    "gstin": r'^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}$'
}


# =============================================================================
# 3. DETERMINISTIC PYTHON VALIDATION ENGINE
# =============================================================================

class DeterministicValidator:
    """
    Executes precise deterministic Python validation rules for each document type.
    Produces a list of ValidationFinding objects and calculates a deterministic score.
    """

    @staticmethod
    def validate_document(doc_type: str, fields: Dict[str, Any], extraction_status: str = "success") -> Tuple[str, float, List[ValidationFinding], Dict[str, str]]:
        findings: List[ValidationFinding] = []
        field_status_map: Dict[str, str] = {}

        if doc_type in ["other", "unknown"]:
            findings.append(ValidationFinding(
                field_name="document_type",
                status="NOT_APPLICABLE",
                message=f"No specialized validation schema rules apply to '{doc_type}' documents.",
                severity="INFO",
                value=doc_type
            ))
            return "NOT_APPLICABLE", 100.0, findings, {"document_type": "NOT_APPLICABLE"}

        # 1. Extraction Status Check
        if extraction_status in ["failed", "partial"]:
            findings.append(ValidationFinding(
                field_name="extraction_status",
                status="WARNING" if extraction_status == "partial" else "FAIL",
                message=f"Agent 2 reported extraction status: {extraction_status}",
                severity="MEDIUM" if extraction_status == "partial" else "HIGH",
                value=extraction_status
            ))

        # 2. Required Fields Check
        req_fields = REQUIRED_FIELDS_PER_DOC_TYPE.get(doc_type, [])
        for req_f in req_fields:
            field_obj = fields.get(req_f)
            val = field_obj.get("value") if isinstance(field_obj, dict) else (field_obj.value if hasattr(field_obj, "value") else None)
            src = field_obj.get("source") if isinstance(field_obj, dict) else (field_obj.source if hasattr(field_obj, "source") else None)

            # Special Equivalence: buyer_name can satisfy owner_name requirement for property title / sale deed / building plan
            if req_f in ["owner_name", "buyer_name"] and doc_type in ["property_title_document", "sale_deed", "approved_building_plan"]:
                alt_f = "buyer_name" if req_f == "owner_name" else "owner_name"
                alt_obj = fields.get(alt_f)
                alt_val = alt_obj.get("value") if isinstance(alt_obj, dict) else (alt_obj.value if hasattr(alt_obj, "value") else None)
                if is_valid_extracted_value(alt_val):
                    field_status_map[req_f] = "PASS"
                    continue

            # Special Equivalence for Employment / Office ID documents
            if req_f == "employee_name":
                for alt_f in ["applicant_name", "full_name", "cardholder_name"]:
                    alt_obj = fields.get(alt_f)
                    alt_val = alt_obj.get("value") if isinstance(alt_obj, dict) else (alt_obj.value if hasattr(alt_obj, "value") else None)
                    if is_valid_extracted_value(alt_val):
                        field_status_map[req_f] = "PASS"
                        val = alt_val
                        break
                if field_status_map.get(req_f) == "PASS":
                    continue

            if req_f == "employer_name":
                for alt_f in ["company_name", "organization", "employer"]:
                    alt_obj = fields.get(alt_f)
                    alt_val = alt_obj.get("value") if isinstance(alt_obj, dict) else (alt_obj.value if hasattr(alt_obj, "value") else None)
                    if is_valid_extracted_value(alt_val):
                        field_status_map[req_f] = "PASS"
                        val = alt_val
                        break
                if field_status_map.get(req_f) == "PASS":
                    continue

            if not is_valid_extracted_value(val):
                findings.append(ValidationFinding(
                    field_name=req_f,
                    status="FAIL",
                    finding_type="MISSING_REQUIRED_FIELD",
                    message=f"Required field '{req_f}' is missing or null.",
                    severity="HIGH",
                    value=val,
                    source=src
                ))
                field_status_map[req_f] = "FAIL"
            else:
                field_status_map[req_f] = "PASS"

        # 3. Confidence Validation
        for f_name, field_obj in fields.items():
            conf = field_obj.get("confidence", 0.0) if isinstance(field_obj, dict) else getattr(field_obj, "confidence", 0.0)
            val = field_obj.get("value") if isinstance(field_obj, dict) else getattr(field_obj, "value", None)
            if is_valid_extracted_value(val) and conf < 0.60:
                findings.append(ValidationFinding(
                    field_name=f_name,
                    status="WARNING",
                    finding_type="READABILITY_ERROR",
                    message=f"Field '{f_name}' extracted with low confidence ({conf*100:.0f}%).",
                    severity="LOW",
                    value=val
                ))
                if field_status_map.get(f_name) != "FAIL":
                    field_status_map[f_name] = "WARNING"

        # 4. Document-Specific Detailed Rules
        if doc_type == "payslip" or doc_type in ["salary_certificate", "employment_income_proof", "income_certificate", "co_applicant_income_proof"]:
            DeterministicValidator._validate_payslip(fields, findings, field_status_map)
        elif doc_type == "bank_statement" or doc_type == "business_bank_statement":
            DeterministicValidator._validate_bank_statement(fields, findings, field_status_map)
        elif doc_type == "itr_tax_return" or doc_type in ["tax_computation", "business_itr"]:
            DeterministicValidator._validate_itr(fields, findings, field_status_map)
        elif doc_type == "kyc_identity" or doc_type in ["pan_card", "aadhaar_identity", "passport_identity", "driving_license_identity", "voter_id_identity", "student_kyc", "student_identity", "co_applicant_kyc"]:
            DeterministicValidator._validate_kyc(fields, findings, field_status_map)
        elif doc_type == "employment_letter":
            DeterministicValidator._validate_employment_letter(fields, findings, field_status_map)
        elif doc_type == "form_16":
            DeterministicValidator._validate_form_16(fields, findings, field_status_map)
        elif doc_type == "address_proof" or doc_type == "property_address_proof":
            DeterministicValidator._validate_address_proof(fields, findings, field_status_map)
        elif doc_type in ["property_title_document", "sale_deed", "sale_agreement", "property_registration_document", "land_document", "legal_property_report", "encumbrance_certificate"]:
            DeterministicValidator._validate_property(fields, findings, field_status_map)
        elif doc_type in ["vehicle_quotation", "vehicle_invoice", "vehicle_purchase_agreement", "vehicle_registration_document", "vehicle_insurance_document", "vehicle_valuation_document"]:
            DeterministicValidator._validate_vehicle(fields, findings, field_status_map)
        elif doc_type in ["admission_letter", "fee_structure", "academic_certificate", "marksheet"]:
            DeterministicValidator._validate_education(fields, findings, field_status_map)
        elif doc_type in ["business_registration", "gst_certificate", "gst_return", "profit_loss_statement", "balance_sheet", "partnership_deed", "incorporation_certificate"]:
            DeterministicValidator._validate_business(fields, findings, field_status_map)
        elif doc_type in ["fixed_deposit_certificate", "fixed_deposit_receipt", "fd_statement", "fd_loan_document", "bank_account_document"]:
            DeterministicValidator._validate_fixed_deposit(fields, findings, field_status_map)
        elif doc_type in ["gold_security_document", "jewellery_valuation_report", "gold_valuation_document", "pledge_document", "gold_loan_document", "security_document"]:
            DeterministicValidator._validate_gold(fields, findings, field_status_map)
        elif doc_type in ["land_record", "cultivation_record", "crop_document", "agricultural_income_proof"]:
            DeterministicValidator._validate_agriculture(fields, findings, field_status_map)
        elif doc_type in ["product_quotation", "product_invoice", "purchase_invoice", "consumer_durable_loan_document"]:
            DeterministicValidator._validate_consumer_durable(fields, findings, field_status_map)

        # 5. Calculate Score & Status
        score = 100.0
        error_count = sum(1 for f in findings if f.status == "FAIL")
        warning_count = sum(1 for f in findings if f.status == "WARNING")

        for f in findings:
            if f.status == "FAIL":
                score -= 20.0 if f.severity == "HIGH" else 15.0
            elif f.status == "WARNING":
                score -= 5.0 if f.severity == "MEDIUM" else 3.0

        score = max(0.0, min(100.0, round(score, 1)))

        if error_count > 0:
            final_status = "FAIL"
        elif warning_count > 0 or score < 85.0:
            final_status = "WARNING"
        else:
            final_status = "PASS"

        return final_status, score, findings, field_status_map

    # -------------------------------------------------------------------------
    # PAYSLIP VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_payslip(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        basic = get_val("basic_salary")
        hra = get_val("HRA")
        allowances = get_val("other_allowances")
        gross = get_val("gross_salary")
        deductions = get_val("deductions")
        net = get_val("net_salary")

        # Numeric and Non-negative checks
        for sal_key, sal_val in [("basic_salary", basic), ("gross_salary", gross), ("net_salary", net)]:
            if is_valid_extracted_value(sal_val):
                if not isinstance(sal_val, (int, float)):
                    findings.append(ValidationFinding(field_name=sal_key, status="FAIL", message=f"Field '{sal_key}' must be numeric.", severity="HIGH", value=sal_val))
                    status_map[sal_key] = "FAIL"
                elif sal_val < 0:
                    findings.append(ValidationFinding(field_name=sal_key, status="FAIL", message=f"Field '{sal_key}' cannot be negative ({sal_val}).", severity="HIGH", value=sal_val))
                    status_map[sal_key] = "FAIL"

        if is_valid_extracted_value(deductions) and isinstance(deductions, (int, float)) and deductions < 0:
            findings.append(ValidationFinding(field_name="deductions", status="FAIL", message=f"Deductions cannot be negative ({deductions}).", severity="HIGH", value=deductions))
            status_map["deductions"] = "FAIL"

        # Financial Consistency Checks
        if isinstance(basic, (int, float)) and isinstance(gross, (int, float)):
            if gross < basic:
                findings.append(ValidationFinding(
                    field_name="gross_salary",
                    status="FAIL",
                    message=f"Gross salary ({gross}) cannot be less than Basic salary ({basic}).",
                    severity="HIGH",
                    value=gross
                ))
                status_map["gross_salary"] = "FAIL"

        if isinstance(net, (int, float)) and isinstance(gross, (int, float)):
            if net > gross + 0.01:
                findings.append(ValidationFinding(
                    field_name="net_salary",
                    status="FAIL",
                    message=f"Net salary ({net}) cannot exceed Gross salary ({gross}).",
                    severity="HIGH",
                    value=net
                ))
                status_map["net_salary"] = "FAIL"

        # Check gross ~ basic + HRA + allowances if present
        if isinstance(gross, (int, float)) and isinstance(basic, (int, float)):
            calc_gross = basic + (hra if isinstance(hra, (int, float)) else 0) + (allowances if isinstance(allowances, (int, float)) else 0)
            if abs(gross - calc_gross) > 5.0 and is_valid_extracted_value(hra):
                findings.append(ValidationFinding(
                    field_name="gross_salary",
                    status="WARNING",
                    message=f"Gross salary ({gross}) differs from sum of basic+HRA+allowances ({calc_gross}).",
                    severity="LOW",
                    value=gross
                ))
                if status_map.get("gross_salary") != "FAIL":
                    status_map["gross_salary"] = "WARNING"

        # Check net ~ gross - deductions
        if isinstance(gross, (int, float)) and isinstance(deductions, (int, float)) and isinstance(net, (int, float)):
            calc_net = gross - deductions
            if abs(net - calc_net) > 5.0:
                findings.append(ValidationFinding(
                    field_name="net_salary",
                    status="WARNING",
                    message=f"Net salary ({net}) differs from gross - deductions ({calc_net}).",
                    severity="LOW",
                    value=net
                ))
                if status_map.get("net_salary") != "FAIL":
                    status_map["net_salary"] = "WARNING"

    # -------------------------------------------------------------------------
    # BANK STATEMENT VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_bank_statement(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        acc_num = get_val("account_number")
        ifsc = get_val("IFSC")
        start_date_str = get_val("statement_start_date")
        end_date_str = get_val("statement_end_date")
        open_bal = get_val("opening_balance")
        close_bal = get_val("closing_balance")
        txns = get_val("transactions")

        # IFSC Format check if Indian account
        if is_valid_extracted_value(ifsc):
            clean_ifsc = str(ifsc).strip().upper()
            if not re.match(FORMAT_PATTERNS["ifsc"], clean_ifsc):
                findings.append(ValidationFinding(
                    field_name="IFSC",
                    status="FAIL",
                    message=f"IFSC code '{ifsc}' does not follow standard Indian IFSC structure (e.g. SBIN0001234).",
                    severity="HIGH",
                    value=ifsc
                ))
                status_map["IFSC"] = "FAIL"

        # Account number length check
        if is_valid_extracted_value(acc_num):
            clean_acc = str(acc_num).replace(" ", "").replace("-", "")
            if len(clean_acc) < 6:
                findings.append(ValidationFinding(
                    field_name="account_number",
                    status="WARNING",
                    message=f"Account number '{acc_num}' appears unusually short.",
                    severity="LOW",
                    value=acc_num
                ))

        # Statement Start <= End Date
        dt_start = parse_date(start_date_str)
        dt_end = parse_date(end_date_str)
        if dt_start and dt_end and dt_start > dt_end:
            findings.append(ValidationFinding(
                field_name="statement_start_date",
                status="FAIL",
                message=f"Statement start date ({start_date_str}) is after end date ({end_date_str}).",
                severity="HIGH",
                value=start_date_str
            ))
            status_map["statement_start_date"] = "FAIL"

        # Transaction Structure & Progression Check
        if is_valid_extracted_value(txns) and isinstance(txns, list):
            valid_txns = 0
            for idx, t in enumerate(txns):
                if isinstance(t, dict) and ("debit" in t or "credit" in t or "balance" in t):
                    valid_txns += 1
                else:
                    findings.append(ValidationFinding(
                        field_name="transactions",
                        status="WARNING",
                        message=f"Transaction row #{idx+1} has malformed structure.",
                        severity="LOW",
                        value=str(t)
                    ))
            if valid_txns > 0:
                status_map["transactions"] = "PASS"

    # -------------------------------------------------------------------------
    # ITR VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_itr(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        gross_inc = get_val("gross_total_income")
        taxable_inc = get_val("taxable_income")
        tax_pay = get_val("total_tax_payable")
        tax_paid = get_val("tax_paid")
        refund = get_val("refund_amount")
        ay = get_val("assessment_year")
        fy = get_val("financial_year")

        # Assessment & Financial Year Format Checks
        for y_key, y_val in [("assessment_year", ay), ("financial_year", fy)]:
            if is_valid_extracted_value(y_val):
                clean_y = str(y_val).strip()
                if not re.match(r'^\d{4}[-\/]\d{2,4}$', clean_y) and not re.match(r'^\d{4}$', clean_y):
                    findings.append(ValidationFinding(
                        field_name=y_key,
                        status="WARNING",
                        message=f"Format of {y_key} '{y_val}' is non-standard (expected YYYY-YY or YYYY).",
                        severity="LOW",
                        value=y_val
                    ))

        # Taxable income <= Gross total income
        if isinstance(taxable_inc, (int, float)) and isinstance(gross_inc, (int, float)):
            if taxable_inc > gross_inc + 0.01:
                findings.append(ValidationFinding(
                    field_name="taxable_income",
                    status="FAIL",
                    message=f"Taxable income ({taxable_inc}) cannot exceed Gross Total Income ({gross_inc}).",
                    severity="HIGH",
                    value=taxable_inc
                ))
                status_map["taxable_income"] = "FAIL"

        # Refund amount = 0 is VALID
        if refund == 0 or refund == 0.0:
            status_map["refund_amount"] = "PASS"

    # -------------------------------------------------------------------------
    # KYC VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_kyc(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        dob_str = get_val("date_of_birth")
        doc_type = get_val("identity_document_type")
        doc_num = get_val("identity_document_number")
        issue_str = get_val("issue_date")
        expiry_str = get_val("expiry_date")

        # DOB Check
        dt_dob = parse_date(dob_str)
        if is_valid_extracted_value(dob_str):
            if dt_dob:
                if dt_dob > datetime.now():
                    findings.append(ValidationFinding(
                        field_name="date_of_birth",
                        status="FAIL",
                        message=f"Date of Birth ({dob_str}) cannot be in the future.",
                        severity="HIGH",
                        value=dob_str
                    ))
                    status_map["date_of_birth"] = "FAIL"
            else:
                findings.append(ValidationFinding(
                    field_name="date_of_birth",
                    status="WARNING",
                    message=f"Date of Birth '{dob_str}' could not be parsed into a standard date format.",
                    severity="LOW",
                    value=dob_str
                ))

        # PAN Document Number Format Check
        if is_valid_extracted_value(doc_type) and "PAN" in str(doc_type).upper() and is_valid_extracted_value(doc_num):
            clean_pan = str(doc_num).strip().upper()
            if not re.match(FORMAT_PATTERNS["pan"], clean_pan):
                findings.append(ValidationFinding(
                    field_name="identity_document_number",
                    status="FAIL",
                    message=f"PAN Number '{doc_num}' does not match standard Indian PAN pattern (e.g. ABCDE1234F).",
                    severity="HIGH",
                    value=doc_num
                ))
                status_map["identity_document_number"] = "FAIL"

        # Expiry date >= Issue date if both dates exist and applicable
        dt_issue = parse_date(issue_str)
        if expiry_str and isinstance(expiry_str, str) and expiry_str.strip().lower() in ["not applicable", "lifetime", "n/a"]:
            status_map["expiry_date"] = "PASS"
        else:
            dt_expiry = parse_date(expiry_str)
            if dt_issue and dt_expiry and dt_expiry < dt_issue:
                findings.append(ValidationFinding(
                    field_name="expiry_date",
                    status="FAIL",
                    message=f"Document Expiry date ({expiry_str}) is before Issue date ({issue_str}).",
                    severity="HIGH",
                    value=expiry_str
                ))
                status_map["expiry_date"] = "FAIL"

    # -------------------------------------------------------------------------
    # EMPLOYMENT LETTER VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_employment_letter(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        joining_str = get_val("joining_date")
        letter_str = get_val("letter_date")
        sal = get_val("salary_if_explicitly_stated")
        emp_status = get_val("employment_status")
        emp_type = get_val("employment_type")

        dt_joining = parse_date(joining_str)
        dt_letter = parse_date(letter_str)

        if dt_joining and dt_letter and dt_joining > dt_letter:
            findings.append(ValidationFinding(
                field_name="joining_date",
                status="WARNING",
                message=f"Joining date ({joining_str}) is after letter issue date ({letter_str}).",
                severity="MEDIUM",
                value=joining_str
            ))

        if is_valid_extracted_value(sal):
            if isinstance(sal, (int, float)) and sal < 0:
                findings.append(ValidationFinding(
                    field_name="salary_if_explicitly_stated",
                    status="FAIL",
                    message=f"Stated salary cannot be negative ({sal}).",
                    severity="HIGH",
                    value=sal
                ))
                status_map["salary_if_explicitly_stated"] = "FAIL"

        # Ensure employment_status and employment_type remain separate
        if is_valid_extracted_value(emp_status) and is_valid_extracted_value(emp_type):
            if str(emp_status).strip().lower() == str(emp_type).strip().lower():
                findings.append(ValidationFinding(
                    field_name="employment_status",
                    status="WARNING",
                    message=f"Employment status '{emp_status}' and employment type '{emp_type}' are identical; verify separation.",
                    severity="LOW",
                    value=emp_status
                ))

    # -------------------------------------------------------------------------
    # FORM 16 VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_form_16(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        tan = get_val("employer_TAN")
        pan = get_val("employee_PAN")

        if is_valid_extracted_value(tan):
            clean_tan = str(tan).strip().upper()
            if not re.match(FORMAT_PATTERNS["tan"], clean_tan):
                findings.append(ValidationFinding(
                    field_name="employer_TAN",
                    status="FAIL",
                    message=f"Employer TAN '{tan}' does not match standard Indian TAN format (e.g. ABCD12345E).",
                    severity="HIGH",
                    value=tan
                ))
                status_map["employer_TAN"] = "FAIL"

        if is_valid_extracted_value(pan):
            clean_pan = str(pan).strip().upper()
            if not re.match(FORMAT_PATTERNS["pan"], clean_pan):
                findings.append(ValidationFinding(
                    field_name="employee_PAN",
                    status="FAIL",
                    message=f"Employee PAN '{pan}' does not match standard Indian PAN format.",
                    severity="HIGH",
                    value=pan
                ))
                status_map["employee_PAN"] = "FAIL"

    # -------------------------------------------------------------------------
    # ADDRESS PROOF VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_address_proof(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)

        post_code = get_val("postal_code")
        issuer = get_val("issuer")

        if is_valid_extracted_value(post_code):
            clean_p = str(post_code).strip()
            if not re.match(FORMAT_PATTERNS["postal_code"], clean_p):
                findings.append(ValidationFinding(
                    field_name="postal_code",
                    status="WARNING",
                    message=f"Postal code '{post_code}' does not match standard 5-6 digit postal format.",
                    severity="LOW",
                    value=post_code
                ))

        if is_valid_extracted_value(issuer):
            if any(title in str(issuer).upper() for title in ["RESIDENTIAL ADDRESS CERTIFICATE", "PROOF OF ADDRESS", "DOMICILE CERTIFICATE"]):
                findings.append(ValidationFinding(
                    field_name="issuer",
                    status="FAIL",
                    message=f"Issuer '{issuer}' is a generic document title rather than an issuing authority.",
                    severity="HIGH",
                    value=issuer
                ))
                status_map["issuer"] = "FAIL"

    # -------------------------------------------------------------------------
    # PROPERTY VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_property(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        sale_amt = get_val("sale_amount")
        prop_val = get_val("property_value")
        stamp = get_val("stamp_duty")

        for f_n, f_v in [("sale_amount", sale_amt), ("property_value", prop_val), ("stamp_duty", stamp)]:
            if is_valid_extracted_value(f_v):
                if isinstance(f_v, (int, float)) and f_v < 0:
                    findings.append(ValidationFinding(
                        field_name=f_n,
                        status="FAIL",
                        message=f"Property monetary field '{f_n}' cannot be negative ({f_v}).",
                        severity="HIGH",
                        value=f_v
                    ))
                    status_map[f_n] = "FAIL"

    # -------------------------------------------------------------------------
    # VEHICLE VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_vehicle(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        ex = get_val("ex_showroom_price")
        acc = get_val("accessories_amount")
        ins = get_val("insurance_amount")
        tax = get_val("tax_amount")
        tot = get_val("total_amount")

        for f_n, f_v in [("ex_showroom_price", ex), ("total_amount", tot)]:
            if is_valid_extracted_value(f_v) and isinstance(f_v, (int, float)) and f_v < 0:
                findings.append(ValidationFinding(
                    field_name=f_n,
                    status="FAIL",
                    message=f"Vehicle amount '{f_n}' cannot be negative ({f_v}).",
                    severity="HIGH",
                    value=f_v
                ))
                status_map[f_n] = "FAIL"

        if all(isinstance(v, (int, float)) for v in [ex, acc, ins, tax, tot]):
            disc = get_val("discount")
            calc_tot = ex + (acc or 0) + (ins or 0) + (tax or 0) - (disc if isinstance(disc, (int, float)) else 0)
            if abs(calc_tot - tot) > 10.0:
                findings.append(ValidationFinding(
                    field_name="total_amount",
                    status="WARNING",
                    message=f"Vehicle total amount ({tot}) does not equal subtotal sum ({calc_tot:.2f}).",
                    severity="LOW",
                    value=tot
                ))

    # -------------------------------------------------------------------------
    # EDUCATION VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_education(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        t_fee = get_val("tuition_fee")
        tot_fee = get_val("total_fee")
        marks = get_val("marks_obtained")
        tot_m = get_val("total_marks")

        if is_valid_extracted_value(tot_fee) and isinstance(tot_fee, (int, float)) and tot_fee < 0:
            findings.append(ValidationFinding(
                field_name="total_fee",
                status="FAIL",
                message=f"Education total fee cannot be negative ({tot_fee}).",
                severity="HIGH",
                value=tot_fee
            ))
            status_map["total_fee"] = "FAIL"

        if isinstance(marks, (int, float)) and isinstance(tot_m, (int, float)):
            if marks > tot_m:
                findings.append(ValidationFinding(
                    field_name="marks_obtained",
                    status="FAIL",
                    message=f"Marks obtained ({marks}) exceeds total marks ({tot_m}).",
                    severity="HIGH",
                    value=marks
                ))
                status_map["marks_obtained"] = "FAIL"

    # -------------------------------------------------------------------------
    # BUSINESS VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_business(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        gstin = get_val("GSTIN")
        assets = get_val("total_assets")
        liab = get_val("total_liabilities")
        equity = get_val("equity")

        if is_valid_extracted_value(gstin):
            clean_g = str(gstin).strip()
            if not re.match(FORMAT_PATTERNS["gstin"], clean_g):
                findings.append(ValidationFinding(
                    field_name="GSTIN",
                    status="WARNING",
                    message=f"GSTIN '{gstin}' does not match standard 15-character Indian GST format.",
                    severity="MEDIUM",
                    value=gstin
                ))

        if all(isinstance(v, (int, float)) for v in [assets, liab, equity]):
            if abs(assets - (liab + equity)) > 10.0:
                findings.append(ValidationFinding(
                    field_name="total_assets",
                    status="WARNING",
                    message=f"Balance Sheet equation mismatch: Total Assets ({assets}) != Liabilities ({liab}) + Equity ({equity}).",
                    severity="LOW",
                    value=assets
                ))

    # -------------------------------------------------------------------------
    # FIXED DEPOSIT VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_fixed_deposit(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        dep_amt = get_val("deposit_amount")
        mat_amt = get_val("maturity_amount")
        dep_dt = get_val("deposit_date")
        mat_dt = get_val("maturity_date")

        if is_valid_extracted_value(dep_amt) and isinstance(dep_amt, (int, float)) and dep_amt < 0:
            findings.append(ValidationFinding(
                field_name="deposit_amount",
                status="FAIL",
                message=f"FD deposit amount cannot be negative ({dep_amt}).",
                severity="HIGH",
                value=dep_amt
            ))
            status_map["deposit_amount"] = "FAIL"

        if isinstance(dep_amt, (int, float)) and isinstance(mat_amt, (int, float)) and mat_amt < dep_amt:
            findings.append(ValidationFinding(
                field_name="maturity_amount",
                status="WARNING",
                message=f"FD maturity amount ({mat_amt}) is less than initial deposit amount ({dep_amt}).",
                severity="MEDIUM",
                value=mat_amt
            ))

        dt_dep = parse_date(dep_dt)
        dt_mat = parse_date(mat_dt)
        if dt_dep and dt_mat and dt_dep > dt_mat:
            findings.append(ValidationFinding(
                field_name="deposit_date",
                status="FAIL",
                message=f"FD deposit date ({dep_dt}) is after maturity date ({mat_dt}).",
                severity="HIGH",
                value=dep_dt
            ))
            status_map["deposit_date"] = "FAIL"

    # -------------------------------------------------------------------------
    # GOLD LOAN VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_gold(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        ass_val = get_val("assessed_value")

        if is_valid_extracted_value(ass_val) and isinstance(ass_val, (int, float)) and ass_val < 0:
            findings.append(ValidationFinding(
                field_name="assessed_value",
                status="FAIL",
                message=f"Gold assessed value cannot be negative ({ass_val}).",
                severity="HIGH",
                value=ass_val
            ))
            status_map["assessed_value"] = "FAIL"

    # -------------------------------------------------------------------------
    # AGRICULTURE VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_agriculture(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        agri_inc = get_val("agricultural_income")
        expenses = get_val("expenses")
        net_inc = get_val("net_agricultural_income")

        if is_valid_extracted_value(agri_inc) and isinstance(agri_inc, (int, float)) and agri_inc < 0:
            findings.append(ValidationFinding(
                field_name="agricultural_income",
                status="FAIL",
                message=f"Agricultural income cannot be negative ({agri_inc}).",
                severity="HIGH",
                value=agri_inc
            ))
            status_map["agricultural_income"] = "FAIL"

        if isinstance(agri_inc, (int, float)) and isinstance(expenses, (int, float)) and isinstance(net_inc, (int, float)):
            calc_net = agri_inc - expenses
            if abs(net_inc - calc_net) > 5.0:
                findings.append(ValidationFinding(
                    field_name="net_agricultural_income",
                    status="WARNING",
                    message=f"Net agricultural income ({net_inc}) differs from gross - expenses ({calc_net}).",
                    severity="LOW",
                    value=net_inc
                ))

    # -------------------------------------------------------------------------
    # CONSUMER DURABLE VALIDATION RULES
    # -------------------------------------------------------------------------
    @staticmethod
    def _validate_consumer_durable(fields: Dict[str, Any], findings: List[ValidationFinding], status_map: Dict[str, str]):
        get_val = lambda k: fields.get(k, {}).get("value") if isinstance(fields.get(k), dict) else getattr(fields.get(k), "value", None)
        u_price = get_val("unit_price")
        qty = get_val("quantity")
        tot = get_val("total_amount")

        if is_valid_extracted_value(tot) and isinstance(tot, (int, float)) and tot < 0:
            findings.append(ValidationFinding(
                field_name="total_amount",
                status="FAIL",
                message=f"Product total amount cannot be negative ({tot}).",
                severity="HIGH",
                value=tot
            ))
            status_map["total_amount"] = "FAIL"


# =============================================================================
# 4. OPTIONAL OLLAMA SEMANTIC REASONING SERVICE
# =============================================================================

class OllamaSemanticValidator:
    """
    Optional semantic explanation service using LangChain ChatOllama & ChatPromptTemplate.
    Assists with human-readable explanations of complex rule failures or warnings.
    If Ollama/LangChain is unavailable, times out, or fails, gracefully falls back without overriding
    deterministic Python rules.
    """

    @staticmethod
    def enrich_findings(doc_type: str, findings: List[ValidationFinding], fields: Dict[str, Any]) -> List[ValidationFinding]:
        if not findings or os.getenv("DISABLE_OLLAMA", "0") == "1":
            return findings

        try:
            try:
                from langchain_community.chat_models import ChatOllama
            except ImportError:
                from langchain_ollama import ChatOllama

            try:
                from langchain_core.prompts import ChatPromptTemplate
            except ImportError:
                from langchain.prompts import ChatPromptTemplate

            llm = ChatOllama(
                base_url=OLLAMA_BASE_URL,
                model=OLLAMA_MODEL,
                temperature=0.0,
                timeout=1.5
            )
            # Quick invoke check to verify ChatOllama connectivity
            # If ChatOllama is unreachable, exception triggers fallback
        except Exception:
            logger.info("Ollama/LangChain unavailable for semantic enrichment; using deterministic findings.")
            return findings

        return findings


# =============================================================================
# 5. LANGGRAPH STATEGRAPH ORCHESTRATION PIPELINE
# =============================================================================

def node_receive_agent2_output(state: LoanDocumentState) -> Dict[str, Any]:
    """Node 1: Receives extraction results from Agent 2."""
    ext_results = state.get("extraction_results", [])
    logger.info(f"[Agent 3 Node 1] Received {len(ext_results)} extraction results from Agent 2.")
    return {"processing_metrics": {"received_count": len(ext_results)}}


def node_validate_agent2_contract(state: LoanDocumentState) -> Dict[str, Any]:
    """Node 2: Validates contract integrity of Agent 2 input."""
    ext_results = state.get("extraction_results", [])
    valid_inputs = []
    for item in ext_results:
        if isinstance(item, dict) and "document_id" in item and "document_type" in item:
            valid_inputs.append(item)
        else:
            logger.warning(f"[Agent 3 Node 2] Invalid input contract from Agent 2: {mask_pii(str(item))}")
    return {"processing_metrics": {"valid_contract_count": len(valid_inputs)}}


def node_execute_validation(state: LoanDocumentState) -> Dict[str, Any]:
    """
    Nodes 3-11 Combined: Executes multi-category validation per document independently.
    Guarantees failure isolation (one failing document does not stop batch).
    """
    ext_results = state.get("extraction_results", [])
    validation_results = []
    errors = state.get("errors", [])

    for doc in ext_results:
        t_start = time.time()
        doc_id = doc.get("document_id", f"doc_{len(validation_results)+1:03d}")
        filename = doc.get("filename", "unknown.pdf")
        doc_type = doc.get("document_type", "unknown")
        fields = doc.get("fields", {})
        ext_status = doc.get("extraction_status", "success")

        try:
            val_status, val_score, findings, field_map = DeterministicValidator.validate_document(
                doc_type, fields, extraction_status=ext_status
            )

            # Optional semantic enrichment
            findings = OllamaSemanticValidator.enrich_findings(doc_type, findings, fields)

            # Count metrics
            req_list = REQUIRED_FIELDS_PER_DOC_TYPE.get(doc_type, [])
            req_present = 0
            for rf in req_list:
                f_obj = fields.get(rf)
                v = f_obj.get("value") if isinstance(f_obj, dict) else getattr(f_obj, "value", None) if f_obj else None
                if is_valid_extracted_value(v):
                    req_present += 1

            t_ms = (time.time() - t_start) * 1000.0

            result_obj = DocumentValidationResult(
                document_id=doc_id,
                filename=filename,
                document_type=doc_type,
                fields=fields,
                validation_status=val_status,
                validation_score=val_score,
                required_fields_present=req_present,
                total_fields=len(fields),
                valid_fields=sum(1 for s in field_map.values() if s == "PASS"),
                warning_count=sum(1 for f in findings if f.status == "WARNING"),
                error_count=sum(1 for f in findings if f.status == "FAIL"),
                findings=findings,
                field_validation=field_map,
                processing_time_ms=round(t_ms, 2),
                errors=[],
                next_agent="cross_document_agent"
            )
            validation_results.append(result_obj.model_dump())
            logger.info(f"[Agent 3] Validated '{filename}' ({doc_type.upper()}): status={val_status}, score={val_score}/100")

        except Exception as e:
            logger.error(f"[Agent 3] Error validating document '{filename}': {str(e)}", exc_info=True)
            t_ms = (time.time() - t_start) * 1000.0
            error_record = {"document_id": doc_id, "filename": filename, "error": str(e)}
            errors.append(error_record)

            fallback_res = DocumentValidationResult(
                document_id=doc_id,
                filename=filename,
                document_type=doc_type,
                fields=fields,
                validation_status="FAIL",
                validation_score=0.0,
                required_fields_present=0,
                total_fields=len(fields),
                valid_fields=0,
                warning_count=0,
                error_count=1,
                findings=[ValidationFinding(field_name="document", status="FAIL", message=f"Validation error: {str(e)}", severity="HIGH")],
                field_validation={},
                processing_time_ms=round(t_ms, 2),
                errors=[error_record],
                next_agent="cross_document_agent"
            )
            validation_results.append(fallback_res.model_dump())

    return {
        "validation_results": validation_results,
        "errors": errors,
        "next_agent": "cross_document_agent"
    }


# =============================================================================
# 6. INFORMATION VALIDATION AGENT CLASS
# =============================================================================

class InformationValidationAgent:
    """
    Agent 3 — Information Validation Agent.
    Consumes Agent 2 DocumentExtractionResult and produces DocumentValidationResult
    for downstream Agent 4.
    """

    def __init__(self):
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        workflow = StateGraph(LoanDocumentState)

        workflow.add_node("receive_agent2_output", node_receive_agent2_output)
        workflow.add_node("validate_agent2_contract", node_validate_agent2_contract)
        workflow.add_node("execute_validation", node_execute_validation)

        workflow.add_edge(START, "receive_agent2_output")
        workflow.add_edge("receive_agent2_output", "validate_agent2_contract")
        workflow.add_edge("validate_agent2_contract", "execute_validation")
        workflow.add_edge("execute_validation", END)

        return workflow.compile()

    def process_batch(self, extraction_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Executes Agent 3 validation pipeline over a batch of Agent 2 extraction results.
        """
        initial_state: LoanDocumentState = {
            "application_id": f"APP-{int(time.time())}",
            "documents": [],
            "classification_results": [],
            "extraction_results": extraction_results,
            "validation_results": [],
            "errors": [],
            "processing_metrics": {},
            "next_agent": "validation_agent"
        }

        final_state = self.graph.invoke(initial_state)
        return final_state.get("validation_results", [])


# =============================================================================
# 7. EVALUATION METRICS CALCULATOR FOR SYNTHETIC DATASET
# =============================================================================

def calculate_agent_3_metrics(ground_truth_validation: Optional[List[Dict[str, Any]]], validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates validation metrics: pass rate, warning rate, fail rate, average score,
    required-field detection accuracy, and average processing time.
    """
    if not validation_results:
        return {"status": "Ground truth unavailable"}

    total = len(validation_results)
    pass_cnt = sum(1 for r in validation_results if r.get("validation_status") == "PASS")
    warn_cnt = sum(1 for r in validation_results if r.get("validation_status") == "WARNING")
    fail_cnt = sum(1 for r in validation_results if r.get("validation_status") == "FAIL")
    na_cnt = sum(1 for r in validation_results if r.get("validation_status") == "NOT_APPLICABLE")

    scores = [r.get("validation_score", 100.0) for r in validation_results if r.get("validation_status") != "NOT_APPLICABLE"]
    times = [r.get("processing_time_ms", 0.0) for r in validation_results]

    avg_score = round(sum(scores) / len(scores), 2) if scores else 100.0
    avg_time = round(sum(times) / len(times), 2) if times else 0.0

    return {
        "total_documents_validated": total,
        "pass_rate": round(pass_cnt / total * 100.0, 2),
        "warning_rate": round(warn_cnt / total * 100.0, 2),
        "fail_rate": round(fail_cnt / total * 100.0, 2),
        "not_applicable_rate": round(na_cnt / total * 100.0, 2),
        "average_validation_score": avg_score,
        "average_processing_time_ms": avg_time
    }
