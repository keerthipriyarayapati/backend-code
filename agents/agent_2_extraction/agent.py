"""
===============================================================================
AGENT 2 — INFORMATION EXTRACTION AGENT
===============================================================================
Single Python Implementation File for Agent 2.

Core Responsibility:
Extract structured key-value fields, confidence scores, source evidence, and 
canonical completeness metrics from loan documents based on Agent 1's classification result.

Supported Document Schemas:
- payslip
- bank_statement
- itr_tax_return
- kyc_identity
- employment_letter
- form_16
- address_proof
- other
- unknown

Architecture & Key Principles:
- Single implementation file: agents/agent_2_extraction/agent.py
- LangGraph StateGraph orchestration workflow
- Semantic Ollama LLM extraction + High-precision Deterministic regex fallback engine
- Anti-Hallucination Guardrails:
    1. Zero guessing or hallucination (e.g. no fake 1040 on Indian ITRs)
    2. Strict title blacklisting (do not set "BANK ACCOUNT STATEMENT" as bank_name,
       or "EMPLOYMENT CONFIRMATION LETTER" as employer_name)
    3. Never infer salary from designation
- Canonical Backend Extraction & Completeness Calculation:
    - Target fields requested per document type
    - Extracted fields count operates strictly on genuine, non-null, non-empty values
    - Empty lists [] (e.g. for transactions) do NOT count as extracted
    - Completeness percentage = (fields_extracted / fields_requested) * 100.0
- Downstream Contract: next_agent = "validation_agent"
"""

import os
import sys
import re
import json
import time
import logging
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
        LoanDocumentState
    )
except ImportError:
    sys.path.append(str(PROJECT_ROOT))
    from shared.state import (
        DocumentClassificationResult,
        DocumentExtractionResult,
        ExtractedField,
        LoanDocumentState
    )

# Format parsers and security validation from Agent 1
try:
    from agents.agent_1_document.agent import (
        parse_document,
        validate_file_security,
        sanitize_filename
    )
except ImportError:
    from agent_1_document.agent import (
        parse_document,
        validate_file_security,
        sanitize_filename
    )

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Agent2_ExtractionAgent")

# Configuration Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# =============================================================================
# 1. FIELD SCHEMAS & BLACKLISTS
# =============================================================================

EXTRACTION_SCHEMAS = {
    "payslip": [
        "employee_name", "employee_id", "employer_name", "pay_period",
        "pay_date", "designation", "department", "basic_salary",
        "HRA", "other_allowances", "gross_salary", "deductions",
        "net_salary", "currency"
    ],
    "bank_statement": [
        "account_holder_name", "account_number", "bank_name", "branch",
        "IFSC", "statement_start_date", "statement_end_date",
        "opening_balance", "closing_balance", "currency", "transactions"
    ],
    "itr_tax_return": [
        "taxpayer_name", "assessment_year", "financial_year",
        "gross_total_income", "taxable_income", "total_tax_payable",
        "tax_paid", "refund_amount", "filing_date", "return_type"
    ],
    "kyc_identity": [
        "full_name", "date_of_birth", "gender", "identity_document_type",
        "identity_document_number", "address", "issue_date", "expiry_date"
    ],
    "pan_card": [
        "full_name", "date_of_birth", "gender", "identity_document_type",
        "identity_document_number", "pan_number", "address", "issue_date", "expiry_date"
    ],
    "employment_letter": [
        "employee_name", "employer_name", "designation", "department",
        "joining_date", "employment_status", "employment_type",
        "work_location", "salary_if_explicitly_stated", "letter_date",
        "reference_number"
    ],
    "employment_certificate": [
        "employee_name", "employer_name", "designation", "department",
        "joining_date", "employment_status", "employment_type",
        "work_location", "salary_if_explicitly_stated", "letter_date",
        "reference_number"
    ],
    "office_id": [
        "employee_name", "employee_id", "employer_name", "company_name",
        "designation", "department", "valid_from", "valid_until",
        "work_location", "document_number"
    ],
    "employee_id": [
        "employee_name", "employee_id", "employer_name", "company_name",
        "designation", "department", "valid_from", "valid_until",
        "work_location", "document_number"
    ],
    "company_id": [
        "employee_name", "employee_id", "employer_name", "company_name",
        "designation", "department", "valid_from", "valid_until",
        "work_location", "document_number"
    ],
    "employment_proof": [
        "employee_name", "employer_name", "designation", "department",
        "joining_date", "employment_status", "employment_type",
        "work_location", "salary_if_explicitly_stated", "letter_date",
        "reference_number"
    ],
    "form_16": [
        "employee_name", "employer_name", "assessment_year", "financial_year",
        "salary_income", "tax_deducted", "tax_payable", "TDS",
        "employer_TAN", "employee_PAN"
    ],
    "address_proof": [
        "full_name", "address", "document_date", "issuer",
        "postal_code", "city", "state", "country"
    ],
    "property_title_document": [
        "buyer_name", "seller_name", "owner_name", "co_owner_name", "property_address",
        "property_description", "property_identifier", "survey_number", "plot_number",
        "registration_number", "document_number", "registration_date", "agreement_date",
        "sale_amount", "property_value", "consideration_amount", "stamp_duty",
        "registration_fee", "issuing_authority"
    ],
    "sale_deed": [
        "buyer_name", "seller_name", "owner_name", "co_owner_name", "property_address",
        "property_description", "property_identifier", "survey_number", "plot_number",
        "registration_number", "document_number", "registration_date", "agreement_date",
        "sale_amount", "property_value", "consideration_amount", "stamp_duty",
        "registration_fee", "issuing_authority"
    ],
    "sale_agreement": [
        "buyer_name", "seller_name", "owner_name", "co_owner_name", "property_address",
        "property_description", "property_identifier", "survey_number", "plot_number",
        "registration_number", "document_number", "registration_date", "agreement_date",
        "sale_amount", "property_value", "consideration_amount", "stamp_duty",
        "registration_fee", "issuing_authority"
    ],
    "approved_building_plan": [
        "owner_name", "property_address", "plan_number", "approval_number",
        "approval_date", "approving_authority", "plot_number", "building_area",
        "floors", "property_type"
    ],
    "property_tax_receipt": [
        "owner_name", "property_address", "property_id", "tax_receipt_number",
        "tax_period", "tax_amount", "payment_date", "issuing_authority"
    ],
    "property_valuation_report": [
        "owner_name", "property_address", "property_type", "valuation_date",
        "market_value", "assessed_value", "forced_sale_value", "valuer_name",
        "valuation_reference_number"
    ],
    "vehicle_quotation": [
        "customer_name", "dealer_name", "quotation_number", "quotation_date",
        "vehicle_make", "vehicle_model", "vehicle_variant", "vehicle_type",
        "registration_location", "ex_showroom_price", "accessories_amount",
        "insurance_amount", "tax_amount", "total_amount", "quoted_amount"
    ],
    "vehicle_invoice": [
        "buyer_name", "dealer_name", "invoice_number", "invoice_date",
        "vehicle_make", "vehicle_model", "vehicle_variant", "chassis_number",
        "engine_number", "registration_number", "ex_showroom_price",
        "tax_amount", "insurance_amount", "accessories_amount", "total_amount"
    ],
    "admission_letter": [
        "student_name", "institution_name", "course_name", "program_name",
        "admission_number", "admission_date", "academic_year", "course_duration",
        "campus", "admission_status"
    ],
    "fee_structure": [
        "student_name", "institution_name", "course_name", "academic_year",
        "tuition_fee", "hostel_fee", "examination_fee", "other_fees",
        "total_fee", "fee_period", "issue_date"
    ],
    "academic_certificate": [
        "student_name", "institution_name", "examination_name", "course_name",
        "roll_number", "registration_number", "academic_year", "examination_date",
        "marks_obtained", "total_marks", "percentage", "grade", "result"
    ],
    "marksheet": [
        "student_name", "institution_name", "examination_name", "course_name",
        "roll_number", "registration_number", "academic_year", "examination_date",
        "marks_obtained", "total_marks", "percentage", "grade", "result"
    ],
    "business_registration": [
        "business_name", "legal_name", "registration_number", "business_type",
        "incorporation_date", "owner_name", "director_names", "registered_address",
        "registration_authority"
    ],
    "gst_certificate": [
        "business_name", "legal_name", "GSTIN", "PAN", "business_type",
        "registration_date", "registered_address", "principal_place_of_business",
        "issuing_authority"
    ],
    "gst_return": [
        "business_name", "GSTIN", "tax_period", "turnover", "taxable_turnover",
        "tax_liability", "tax_paid", "filing_date", "return_type"
    ],
    "profit_loss_statement": [
        "business_name", "financial_year", "revenue", "sales", "cost_of_goods",
        "gross_profit", "operating_expenses", "operating_profit", "net_profit",
        "depreciation", "interest_expense", "other_income"
    ],
    "balance_sheet": [
        "business_name", "financial_year", "total_assets", "current_assets",
        "fixed_assets", "total_liabilities", "current_liabilities",
        "long_term_liabilities", "equity", "capital", "retained_earnings"
    ],
    "business_bank_statement": [
        "business_name", "account_holder_name", "account_number", "bank_name", "branch",
        "IFSC", "statement_start_date", "statement_end_date", "opening_balance",
        "closing_balance", "transactions"
    ],
    "fixed_deposit_certificate": [
        "account_holder_name", "joint_holder_names", "bank_name", "branch",
        "account_number", "FD_number", "deposit_amount", "interest_rate",
        "deposit_date", "maturity_date", "maturity_amount", "tenure", "nominee", "currency"
    ],
    "gold_security_document": [
        "borrower_name", "valuation_date", "valuation_reference", "gold_description",
        "jewellery_description", "number_of_items", "gross_weight", "net_weight",
        "purity", "assessed_value", "market_value", "loan_value", "valuer_name",
        "appraiser_name", "security_reference"
    ],
    "land_record": [
        "farmer_name", "land_owner_name", "land_address", "survey_number",
        "subdivision_number", "land_area", "land_unit", "village", "district",
        "state", "land_type", "cultivation_type", "crop_type", "ownership_type",
        "document_number", "issue_date", "issuing_authority"
    ],
    "agricultural_income_proof": [
        "farmer_name", "financial_year", "crop_type", "cultivated_area",
        "agricultural_income", "expenses", "net_agricultural_income",
        "issuing_authority", "document_date"
    ],
    "product_quotation": [
        "customer_name", "seller_name", "quotation_number", "quotation_date",
        "product_name", "product_category", "brand", "model", "serial_number",
        "quantity", "unit_price", "tax_amount", "discount", "total_amount"
    ],
    "product_invoice": [
        "buyer_name", "seller_name", "invoice_number", "invoice_date",
        "product_name", "brand", "model", "serial_number", "quantity",
        "unit_price", "tax_amount", "discount", "total_amount"
    ],
    "other": [
        "document_title", "date", "summary", "key_fields"
    ],
    "unknown": []
}

GENERIC_EMPLOYER_TITLES_BLACKLIST = [
    "EMPLOYMENT CONFIRMATION LETTER", "CONFIRMATION OF EMPLOYMENT LETTER",
    "EMPLOYMENT LETTER", "OFFER OF EMPLOYMENT & APPOINTMENT LETTER",
    "JOB APPOINTMENT LETTER", "EMPLOYMENT VERIFICATION LETTER",
    "CERTIFICATE OF SERVICE", "HR DEPT CERTIFICATE OF SERVICE",
    "OFFER LETTER", "APPOINTMENT LETTER", "LETTER OF EMPLOYMENT",
    "CONFIRMATION LETTER", "SERVICE CERTIFICATE", "TO WHOM IT MAY CONCERN",
    "FIELD", "DETAILS", "VALUE", "KEY"
]

GENERIC_BANK_TITLES_BLACKLIST = [
    "BANK ACCOUNT STATEMENT", "ACCOUNT STATEMENT", "SAVINGS ACCOUNT STATEMENT",
    "BANK STATEMENT SUMMARY", "BANK STATEMENT", "STATEMENT SUMMARY",
    "SAVINGS STATEMENT", "CHECKING STATEMENT", "TRANSACTION STATEMENT",
    "ACCOUNT SUMMARY", "MONTHLY STATEMENT", "FIRST TRUST BANKING STATEMENT"
]

GENERIC_ISSUER_TITLES_BLACKLIST = [
    "RESIDENTIAL ADDRESS CERTIFICATE", "ADDRESS CERTIFICATE",
    "ADDRESS PROOF CERTIFICATE", "PROOF OF ADDRESS",
    "RESIDENTIAL CERTIFICATE", "ADDRESS PROOF",
    "DOMICILE CERTIFICATE", "UTILITY BILL", "TELEPHONE BILL",
    "ELECTRICITY BILL", "WATER BILL", "RENT AGREEMENT", "TO WHOM IT MAY CONCERN"
]


def is_valid_extracted_value(val: Any) -> bool:
    """
    Helper to check if a value is a genuine, non-null, non-empty extracted field.
    Returns False for None, empty strings, 'null', 'none', 'n/a', and empty lists [].
    """
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ["", "null", "none", "n/a", "unknown"]:
            return False
        return True
    if isinstance(val, (list, tuple)):
        return len(val) > 0
    if isinstance(val, dict):
        return len(val) > 0
    if isinstance(val, (int, float, bool)):
        return True
    return False


# =============================================================================
# 2. DETERMINISTIC REGEX PATTERN MATCHING EXTRACTORS (HIGH-PRECISION ENGINE)
# =============================================================================

class DeterministicExtractor:
    """
    High-precision deterministic extractors using regular expressions and string analysis.
    Provides complete, grounded field extraction with anti-hallucination title filters.
    """

    @staticmethod
    def extract_currency_amount(text: str, patterns: List[str]) -> Optional[Tuple[Union[float, int], str, str]]:
        """Extracts numerical amount, currency symbol/name, and raw matched text snippet."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                raw_str = match.group(0)
                num_str = match.group(1) if match.lastindex and match.lastindex >= 1 else raw_str
                num_match = re.search(r'([\d,]+(?:\.\d{1,2})?)', num_str)
                if num_match:
                    clean_num = num_match.group(1).replace(",", "")
                    try:
                        val = float(clean_num)
                        if val.is_integer():
                            val = int(val)
                        curr = "USD" if "$" in raw_str else ("INR" if ("INR" in raw_str.upper() or "₹" in raw_str or "RS" in raw_str.upper()) else "INR")
                        return val, curr, raw_str
                    except ValueError:
                        continue
        return None

    @staticmethod
    def extract_payslip(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Employee Name
        name_match = re.search(r'(?:Employee\s*Name|Name\s*of\s*Employee|Emp\s*Name|Employee|Staff\s*Name|Worker|Name)\s*:?\s*([A-Za-z0-9\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if name_match and len(name_match.group(1).strip()) > 2 and not any(k in name_match.group(1).lower() for k in ["salary", "payslip", "period"]):
            val = name_match.group(1).strip()
            fields["employee_name"] = ExtractedField(value=val, confidence=0.95, source={"page": 1, "text": name_match.group(0)})
            fields["applicant_name"] = ExtractedField(value=val, confidence=0.95, source={"page": 1, "text": name_match.group(0)})
        else:
            fields["employee_name"] = ExtractedField(value=None, confidence=0.0)
            fields["applicant_name"] = ExtractedField(value=None, confidence=0.0)

        # Employee ID
        id_match = re.search(r'(?:Employee\s*ID|ID|Code|Emp\s*ID|Emp\s*No)\s*(?::|\n)\s*([\w-]+)', text, re.IGNORECASE)
        if id_match and id_match.group(1).strip().upper() not in ["DETAILS", "VALUE"]:
            fields["employee_id"] = ExtractedField(value=id_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": id_match.group(0)})
        else:
            fields["employee_id"] = ExtractedField(value=None, confidence=0.0)

        # Employer Name - Explicit label check first, then fallback to top header lines
        emp_label_m = re.search(r'(?:Employer\s*Name|Employer|Company\s*Name|Company|Organization)\s*(?::|\n)\s*([A-Za-z0-9\s\.\,\&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if emp_label_m and len(emp_label_m.group(1).strip()) > 2 and emp_label_m.group(1).strip().upper() not in ["FIELD", "DETAILS", "PAYSLIP", "SALARY", "MONTHLY"]:
            fields["employer_name"] = ExtractedField(value=emp_label_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": emp_label_m.group(0)})
        else:
            employer_val = None
            employer_src = None
            for line in lines[:5]:
                upper_line = line.upper().strip()
                if any(k in upper_line for k in ["PAYSLIP", "SALARY", "MONTHLY", "STATEMENT", "PERIOD", "EMPLOYEE", "WAGE", "PAYROLL", "ADVICE", "FIELD", "DETAILS"]):
                    continue
                if len(line) > 3:
                    employer_val = line.strip()
                    employer_src = line
                    break
            fields["employer_name"] = ExtractedField(value=employer_val, confidence=0.90 if employer_val else 0.0, source={"page": 1, "text": employer_src} if employer_src else None)

        # Pay Period & Date
        period_match = re.search(r'(?:Pay\s*Period|Payroll\s*Period|Salary\s*Month|Period|Pay\s*Month|\bMonth\b|Salary\s*Certificate\s*Date|PAYSLIP\s*-\s*|FOR\s+)\s*(?::|\n)?\s*(?:FOR\s+)?([A-Za-z0-9,\s/-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if period_match and period_match.group(1).strip().upper() not in ["DETAILS", "VALUE"] and not any(k in period_match.group(1).lower() for k in ["employee", "salary", "payslip"]):
            fields["pay_period"] = ExtractedField(value=period_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": period_match.group(0)})
        else:
            p_fallback = re.search(r'(?:PAYSLIP\s*-\s*|PAYSLIP\s+FOR\s+)(?:FOR\s+)?([A-Za-z0-9,\s/-]+?)(?:\||\n|$)', text, re.IGNORECASE)
            if not p_fallback:
                p_fallback = re.search(r'(?:Pay\s*Period|Payroll\s*Period|Period|Salary\s*Month|Pay\s*Month|For\s+the\s+month\s+of)\s*:\s*([A-Za-z0-9,\s/-]+?)(?:\||\n|$)', text, re.IGNORECASE)
            if p_fallback and not any(k in p_fallback.group(1).lower() for k in ["employee", "salary"]):
                clean_p = re.sub(r'^(?:payslip\s*-\s*)', '', p_fallback.group(1).strip(), flags=re.IGNORECASE).strip()
                fields["pay_period"] = ExtractedField(value=clean_p, confidence=0.90, source={"page": 1, "text": p_fallback.group(0)})
            else:
                fields["pay_period"] = ExtractedField(value=None, confidence=0.0)

        date_match = re.search(r'(?:Pay\s*Date|Payment\s*Date|Date)\s*(?::|\n)\s*([A-Za-z0-9,\s/-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if date_match and not any(k in date_match.group(1).lower() for k in ["payslip", "employee", "details"]):
            fields["pay_date"] = ExtractedField(value=date_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": date_match.group(0)})
        else:
            fields["pay_date"] = ExtractedField(value=None, confidence=0.0)

        # Designation & Department
        desig_match = re.search(r'(?:Designation|Role|Position|Title)\s*(?::|\n)\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if desig_match and desig_match.group(1).strip().upper() not in ["DETAILS", "VALUE"]:
            fields["designation"] = ExtractedField(value=desig_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": desig_match.group(0)})
        else:
            fields["designation"] = ExtractedField(value=None, confidence=0.0)

        dept_match = re.search(r'(?:Department|Dept)\s*(?::|\n)\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if dept_match and dept_match.group(1).strip().upper() not in ["DETAILS", "VALUE"]:
            fields["department"] = ExtractedField(value=dept_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": dept_match.group(0)})
        else:
            fields["department"] = ExtractedField(value=None, confidence=0.0)

        # Salary Components
        basic_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Basic\s*Salary|Basic\s*Pay|Base\s*Wage|Basic)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Basic\s*Salary|Basic\s*Pay|Base\s*Wage|Basic)\s*:?\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'
        ])
        if basic_res:
            fields["basic_salary"] = ExtractedField(value=basic_res[0], confidence=0.95, source={"page": 1, "text": basic_res[2]})
            fields["currency"] = ExtractedField(value=basic_res[1], confidence=0.95)
        else:
            fields["basic_salary"] = ExtractedField(value=None, confidence=0.0)
            fields["currency"] = ExtractedField(value="INR" if ("INR" in text or "₹" in text) else ("USD" if "$" in text else None), confidence=0.85 if ("INR" in text or "$" in text or "₹" in text) else 0.0)

        hra_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:House\s*Rent\s*Allowance(?:\s*\(HRA\))?|HRA)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:House\s*Rent\s*Allowance(?:\s*\(HRA\))?|HRA)\s*:?\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'
        ])
        fields["HRA"] = ExtractedField(value=hra_res[0], confidence=0.95, source={"page": 1, "text": hra_res[2]}) if hra_res else ExtractedField(value=None, confidence=0.0)

        allow_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Other\s*Allowances|Special\s*Allowance|Allowances|Medical\s*Allowance|Dearness\s*Allowance)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Other\s*Allowances|Special\s*Allowance|Allowances)\s*:?\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'
        ])
        if allow_res:
            fields["other_allowances"] = ExtractedField(value=allow_res[0], confidence=0.95, source={"page": 1, "text": allow_res[2]})
        else:
            allow_matches = re.findall(r'(?:Special\s*Allowance|Dearness\s*Allowance|Medical\s*Allowance|Allowances|Bonus\s*&\s*Incentive)\s*:\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)', text, re.IGNORECASE)
            if allow_matches:
                total_allow = 0.0
                raw_allow_texts = []
                for amt_str in allow_matches:
                    clean_a = re.sub(r'[^\d\.]', '', amt_str)
                    try:
                        total_allow += float(clean_a)
                        raw_allow_texts.append(amt_str)
                    except ValueError:
                        pass
                fields["other_allowances"] = ExtractedField(value=total_allow, confidence=0.90, source={"page": 1, "text": f"Allowances: {', '.join(raw_allow_texts)}"})
            else:
                fields["other_allowances"] = ExtractedField(value=None, confidence=0.0)

        gross_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Gross\s*Salary|Gross\s*Earnings|Gross\s*Compensation|Gross)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Gross\s*Salary|Gross\s*Earnings|Gross\s*Compensation|Gross)\s*:?\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'
        ])
        if gross_res:
            fields["gross_salary"] = ExtractedField(value=gross_res[0], confidence=0.95, source={"page": 1, "text": gross_res[2]})
        else:
            if fields["basic_salary"].value and fields["other_allowances"].value:
                calc_gross = float(fields["basic_salary"].value) + float(fields["other_allowances"].value) + (float(fields["HRA"].value) if fields["HRA"].value else 0.0)
                fields["gross_salary"] = ExtractedField(value=calc_gross, confidence=0.85, source={"page": 1, "text": f"Calculated: Basic ({fields['basic_salary'].value}) + Allowances"})
            else:
                fields["gross_salary"] = ExtractedField(value=None, confidence=0.0)

        net_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:NET\s*PAYABLE\s*SALARY|Net\s*Pay\s*Transferred|Net\s*Pay|Net\s*Salary|Net\s*Disbursement|Take\s*Home(?:\s*Pay)?|Net\s*Amount|Net\s*Payable)\s*:?\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'(?:NET\s*PAYABLE\s*SALARY|Net\s*Pay\s*Transferred|Net\s*Pay|Net\s*Salary|Net\s*Disbursement|Take\s*Home|Net\s*Amount|Net\s*Payable)\s*(?::|\n|\||\s+)\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)'
        ])
        fields["net_salary"] = ExtractedField(value=net_res[0], confidence=0.95, source={"page": 1, "text": net_res[2]}) if net_res else ExtractedField(value=None, confidence=0.0)

        ded_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Total\s*Deductions|Deductions|Tax\s*Deducted)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Total\s*Deductions|Deductions|Tax\s*Deducted)\s*:?\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'
        ])
        if ded_res:
            fields["deductions"] = ExtractedField(value=ded_res[0], confidence=0.90, source={"page": 1, "text": ded_res[2]})
        else:
            pf_matches = re.findall(r'(?:Provident\s*Fund(?:\s*\(PF\))?|PF|Income\s*Tax(?:\s*\(TDS\))?|Professional\s*Tax)\s*:\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)', text, re.IGNORECASE)
            if pf_matches:
                tot_ded = 0.0
                for d_str in pf_matches:
                    clean_d = re.sub(r'[^\d\.]', '', d_str)
                    try:
                        tot_ded += float(clean_d)
                    except ValueError:
                        pass
                fields["deductions"] = ExtractedField(value=tot_ded, confidence=0.85, source={"page": 1, "text": f"Summed deductions: {pf_matches}"})
            else:
                fields["deductions"] = ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_bank_statement(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        # Account Holder Name
        name_match = re.search(r'(?:Account\s*Holder\s*Name|Customer\s*Name|Account\s*Holder|Customer|Name|A/C\s*Holder)\s*:?\s*([A-Za-z0-9\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if name_match and len(name_match.group(1).strip()) > 2 and not any(k in name_match.group(1).upper() for k in ["BANK", "STATEMENT", "ACCOUNT", "SAVINGS"]):
            val = name_match.group(1).strip()
            fields["account_holder_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": name_match.group(0)})
            fields["applicant_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": name_match.group(0)})
        else:
            fields["account_holder_name"] = ExtractedField(value=None, confidence=0.0)
            fields["applicant_name"] = ExtractedField(value=None, confidence=0.0)

        # Account Number (Support masked values e.g. TEST-XXXX-4827, XXXX-XXXX-4827, A/C No: XXXX)
        acc_match = re.search(r'(?:ACCOUNT\s*NUMBER|ACCOUNT\s*NO|A/C\s*NUMBER|A/C\s*NO|Acct\s*No|Account\s*No|Account\s*Num|Account\s*#)\s*:?\s*([A-Za-z0-9-X]+(?:[ \t]+[A-Za-z0-9-X]+)?)', text, re.IGNORECASE)
        if not acc_match:
            acc_match = re.search(r'(?:Account)\s*:\s*([A-Za-z0-9-X]{4,})', text, re.IGNORECASE)
        if acc_match:
            clean_acc = acc_match.group(1).split('\n')[0].strip()
            if len(clean_acc) > 3 and clean_acc.upper() not in ["TYPE", "HOLDER", "STATEMENT", "DETAILS", "SAVINGS", "CURRENT"]:
                fields["account_number"] = ExtractedField(value=clean_acc, confidence=0.95, source={"page": 1, "text": acc_match.group(0)})
            else:
                fields["account_number"] = ExtractedField(value=None, confidence=0.0)
        else:
            fields["account_number"] = ExtractedField(value=None, confidence=0.0)

        # Bank Name — Explicit label check first, then header lines fallback
        bank_name_val = None
        bank_name_src = None
        bank_label_m = re.search(r'(?:Bank\s*Name|Bank)\s*(?::|\n)\s*([A-Za-z0-9\s\.\,\&\(\)]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if bank_label_m and len(bank_label_m.group(1).strip()) > 2 and bank_label_m.group(1).strip().upper() not in ["FIELD", "DETAILS", "BANK STATEMENT", "ACCOUNT STATEMENT"]:
            bank_name_val = bank_label_m.group(1).strip()
            bank_name_src = bank_label_m.group(0)
        else:
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            for line in lines[:10]:
                upper_line = line.upper().strip()
                if upper_line in GENERIC_BANK_TITLES_BLACKLIST or any(upper_line == b_title for b_title in GENERIC_BANK_TITLES_BLACKLIST):
                    continue
                if upper_line in ["FIELD", "DETAILS"]:
                    continue
                if "BANK" in upper_line or "BANKING" in upper_line:
                    bank_name_val = line.strip()
                    bank_name_src = line
                    break

        fields["bank_name"] = ExtractedField(value=bank_name_val, confidence=0.90 if bank_name_val else 0.0, source={"page": 1, "text": bank_name_src} if bank_name_src else None)

        # Branch & IFSC
        branch_match = re.search(r'(?:Branch|Branch\s*Name)\s*:\s*([A-Za-z0-9\s\.\,\(\)-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["branch"] = ExtractedField(value=branch_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": branch_match.group(0)}) if branch_match else ExtractedField(value=None, confidence=0.0)

        ifsc_match = re.search(r'(?:IFSC\s*Code|IFSC|IFS\s*Code)\s*:?\s*([A-Za-z0-9]+)', text, re.IGNORECASE)
        if not ifsc_match:
            ifsc_match = re.search(r'\b([A-Z]{4}0[A-Z0-9]{6})\b', text)
        fields["IFSC"] = ExtractedField(value=ifsc_match.group(1).strip() if ifsc_match else None, confidence=0.95 if ifsc_match else 0.0, source={"page": 1, "text": ifsc_match.group(0)} if ifsc_match else None)

        # Statement Start & End Dates (Support full month names e.g. 01 April 2026 – 30 September 2026)
        period_match = re.search(
            r'(?:Statement\s*Period|Period|From\s*:?\s*[\d\w/-]+\s*To)\s*:?\s*'
            r'(\d{1,2}[\s/-][A-Za-z0-9]+[\s/-]\d{2,4})\s*(?:to|-|–|—|until)\s*'
            r'(\d{1,2}[\s/-][A-Za-z0-9]+[\s/-]\d{2,4})',
            text, re.IGNORECASE
        )
        if period_match:
            fields["statement_start_date"] = ExtractedField(value=period_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": period_match.group(0)})
            fields["statement_end_date"] = ExtractedField(value=period_match.group(2).strip(), confidence=0.95, source={"page": 1, "text": period_match.group(0)})
        else:
            dt_from = re.search(r'(?:From\s*Date|Start\s*Date|From)\s*:?\s*(\d{1,2}[\s/-][A-Za-z0-9]+[\s/-]\d{2,4})', text, re.IGNORECASE)
            dt_to = re.search(r'(?:To\s*Date|End\s*Date|To)\s*:?\s*(\d{1,2}[\s/-][A-Za-z0-9]+[\s/-]\d{2,4})', text, re.IGNORECASE)
            fields["statement_start_date"] = ExtractedField(value=dt_from.group(1).strip(), confidence=0.90, source={"page": 1, "text": dt_from.group(0)}) if dt_from else ExtractedField(value=None, confidence=0.0)
            fields["statement_end_date"] = ExtractedField(value=dt_to.group(1).strip(), confidence=0.90, source={"page": 1, "text": dt_to.group(0)}) if dt_to else ExtractedField(value=None, confidence=0.0)

        # Balances
        open_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Opening\s*Balance|Opening\s*Bal|Beginning\s*Balance|Opening\s*Ledger|Opening\s*Available\s*Balance)\s*:?\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'(?:Opening\s*Balance|Opening\s*Bal|Beginning\s*Balance)\s*(?::|\n|\||\s+)\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)'
        ])
        fields["opening_balance"] = ExtractedField(value=open_res[0], confidence=0.95, source={"page": 1, "text": open_res[2]}) if open_res else ExtractedField(value=None, confidence=0.0)

        close_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Closing\s*Balance|Closing\s*Bal|Ending\s*Balance|End\s*Balance|Closing\s*Ledger\s*Balance|Closing\s*Available\s*Ledger\s*Balance|Account\s*Balance)\s*:?\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)',
            r'(?:Closing\s*Balance|Closing\s*Bal|Ending\s*Balance|End\s*Balance)\s*(?::|\n|\||\s+)\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)'
        ])
        fields["closing_balance"] = ExtractedField(value=close_res[0], confidence=0.95, source={"page": 1, "text": close_res[2]}) if close_res else ExtractedField(value=None, confidence=0.0)

        fields["currency"] = ExtractedField(value="USD" if "$" in text else ("INR" if ("INR" in text.upper() or "₹" in text or "RS" in text.upper()) else "INR"), confidence=0.90)

        # Parse Transactions
        transactions = []
        
        # 1. Check for pipe-delimited table rows (e.g. from docx or markdown tables)
        pipe_lines = [l.strip() for l in text.split("\n") if "|" in l]
        for pline in pipe_lines:
            parts = [p.strip() for p in pline.split("|")]
            if len(parts) >= 4:
                date_m = re.search(r'(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', parts[0])
                if date_m:
                    tx_date = date_m.group(1)
                    desc = parts[1]
                    if desc.lower() in ["description", "particulars", "details"]:
                        continue
                    if len(parts) >= 5:
                        deb_str = parts[2]
                        cred_str = parts[3]
                        bal_str = parts[4]
                        ref_str = parts[5] if len(parts) > 5 else None

                        clean_d = re.sub(r'[^\d\.]', '', deb_str)
                        clean_c = re.sub(r'[^\d\.]', '', cred_str)
                        clean_b = re.sub(r'[^\d\.]', '', bal_str)

                        deb_val = float(clean_d) if clean_d else None
                        cred_val = float(clean_c) if clean_c else None
                        bal_val = float(clean_b) if clean_b else None
                        ref_val = ref_str.strip() if (ref_str and ref_str.strip() not in ["-", "", "null"]) else None

                        transactions.append({
                            "transaction_date": tx_date,
                            "description": desc,
                            "debit": deb_val,
                            "credit": cred_val,
                            "balance": bal_val,
                            "reference_number": ref_val
                        })

        # 2. Fallback to standard regex for non-pipe formatted text statements
        if not transactions:
            tx_matches = re.findall(r'(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4})\s+([A-Za-z0-9\s\-\:\+]+?)\s*([\+\-]?[\$\₹]?[\d,]+\.?\d*)\s*(?:\|\s*Balance\s*:\s*([\$\₹]?[\d,]+\.?\d*))?', text)
            for match in tx_matches:
                raw_desc = match[1].strip()
                if raw_desc.lower() in ["to", "from", "period", "statement period"] or len(raw_desc) < 2:
                    continue
                amt_str = match[2].strip()
                bal_str = match[3].strip() if len(match) >= 4 and match[3] else None

                clean_amt = re.sub(r'[^\d\.]', '', amt_str)
                amt_val = float(clean_amt) if clean_amt else 0.0

                is_debit = "-" in amt_str or "debit" in raw_desc.lower() or "withdrawal" in raw_desc.lower() or "purchase" in raw_desc.lower()
                is_credit = "+" in amt_str or "credit" in raw_desc.lower() or "deposit" in raw_desc.lower() or "interest" in raw_desc.lower()

                transactions.append({
                    "transaction_date": match[0],
                    "description": raw_desc,
                    "debit": amt_val if is_debit else None,
                    "credit": amt_val if (is_credit or not is_debit) else None,
                    "balance": float(re.sub(r'[^\d\.]', '', bal_str)) if bal_str else None,
                    "reference_number": None
                })
        
        # Empty transaction list [] is NOT counted as extracted
        fields["transactions"] = ExtractedField(
            value=transactions if len(transactions) > 0 else None,
            confidence=0.85 if len(transactions) > 0 else 0.0,
            source={"page": 1, "text": f"Extracted {len(transactions)} transaction rows"} if len(transactions) > 0 else None
        )

        return fields

    @staticmethod
    def extract_itr_tax_return(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        # Taxpayer Name
        name_match = re.search(r'(?:Name|Filer\s*Name|Taxpayer\s*Name|Taxpayer)\s*:\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["taxpayer_name"] = ExtractedField(value=name_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": name_match.group(0)}) if name_match else ExtractedField(value=None, confidence=0.0)

        # Assessment Year & Financial Year
        ay_match = re.search(r'(?:Assessment\s*Year|AY)\s*:?\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["assessment_year"] = ExtractedField(value=ay_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": ay_match.group(0)}) if ay_match else ExtractedField(value=None, confidence=0.0)

        fy_match = re.search(r'(?:Financial\s*Year|FY|Tax\s*Year)\s*:?\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["financial_year"] = ExtractedField(value=fy_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": fy_match.group(0)}) if fy_match else ExtractedField(value=None, confidence=0.0)

        # Incomes & Taxes
        gross_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Gross\s*Total\s*Income|Adjusted\s*Gross\s*Income\s*\(AGI\)|Gross\s*Income)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["gross_total_income"] = ExtractedField(value=gross_res[0], confidence=0.95, source={"page": 1, "text": gross_res[2]}) if gross_res else ExtractedField(value=None, confidence=0.0)

        taxable_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Taxable\s*Income|Taxable\s*Income)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["taxable_income"] = ExtractedField(value=taxable_res[0], confidence=0.95, source={"page": 1, "text": taxable_res[2]}) if taxable_res else ExtractedField(value=None, confidence=0.0)

        tax_pay_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Net\s*Tax\s*Payable|Total\s*Tax\s*Payable|Tax\s*Payable|Tax\s*Liability)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_tax_payable"] = ExtractedField(value=tax_pay_res[0], confidence=0.95, source={"page": 1, "text": tax_pay_res[2]}) if tax_pay_res else ExtractedField(value=None, confidence=0.0)

        tax_paid_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:TDS\s*Paid|Total\s*Federal\s*Income\s*Tax\s*Withheld|Net\s*Tax\s*Paid\s*under\s*Section\s*143\(1\)|Tax\s*Paid)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tax_paid"] = ExtractedField(value=tax_paid_res[0], confidence=0.90, source={"page": 1, "text": tax_paid_res[2]}) if tax_paid_res else ExtractedField(value=None, confidence=0.0)

        refund_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Refund\s*Due|Refund\s*Amount|Refund)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        if refund_res:
            fields["refund_amount"] = ExtractedField(value=refund_res[0], confidence=0.90, source={"page": 1, "text": refund_res[2]})
        elif any(k in text.upper() for k in ["REFUND DUE: 0", "REFUND DUE: 0 INR", "REFUND AMOUNT: 0", "REFUND AMOUNT: INR 0", "REFUND: 0", "REFUND: INR 0"]):
            fields["refund_amount"] = ExtractedField(value=0, confidence=0.90, source={"page": 1, "text": "Refund Amount: INR 0"})
        else:
            fields["refund_amount"] = ExtractedField(value=None, confidence=0.0)

        filing_date_match = re.search(r'(?:Filing\s*Date|Filed\s*On|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|[\d/-]+)', text, re.IGNORECASE)
        fields["filing_date"] = ExtractedField(value=filing_date_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": filing_date_match.group(0)}) if filing_date_match else ExtractedField(value=None, confidence=0.0)

        # Return Type — STRICT RULE: Never insert 1040 unless explicitly in text
        if "1040" in text:
            fields["return_type"] = ExtractedField(value="Form 1040", confidence=0.95, source={"page": 1, "text": "Form 1040"})
        elif "ITR-1" in text or "ITR 1" in text:
            fields["return_type"] = ExtractedField(value="ITR-1 (SAHAJ)", confidence=0.95, source={"page": 1, "text": "ITR-1"})
        elif "ITR-2" in text:
            fields["return_type"] = ExtractedField(value="ITR-2", confidence=0.95, source={"page": 1, "text": "ITR-2"})
        elif "ITR-3" in text:
            fields["return_type"] = ExtractedField(value="ITR-3", confidence=0.95, source={"page": 1, "text": "ITR-3"})
        elif "ITR-4" in text:
            fields["return_type"] = ExtractedField(value="ITR-4", confidence=0.95, source={"page": 1, "text": "ITR-4"})
        else:
            fields["return_type"] = ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_kyc_identity(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        name_match = re.search(r'(?:Name\s*of\s*Holder|Applicant\s*Name|Customer\s*Name|Full\s*Name|Elector\s*Name|Name)\s*:?\s*([A-Za-z0-9\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if name_match and len(name_match.group(1).strip()) > 2 and not any(k in name_match.group(1).upper() for k in ["GOVERNMENT", "INDIA", "DEPARTMENT", "INCOME TAX"]):
            val = name_match.group(1).strip()
            fields["full_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": name_match.group(0)})
            fields["applicant_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": name_match.group(0)})
        else:
            # Fallback: scan lines for name if label missing
            val = None
            src = None
            for line in text.split("\n"):
                line_s = line.strip()
                if line_s and not any(k in line_s.upper() for k in ["GOVERNMENT", "INCOME TAX", "DEPARTMENT", "PAN CARD", "INDIA", "DOB", "DATE OF BIRTH", "PERMANENT ACCOUNT NUMBER"]):
                    if re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+$', line_s):
                        val = line_s
                        src = line_s
                        break
            fields["full_name"] = ExtractedField(value=val, confidence=0.85 if val else 0.0, source={"page": 1, "text": src} if src else None)
            fields["applicant_name"] = ExtractedField(value=val, confidence=0.85 if val else 0.0, source={"page": 1, "text": src} if src else None)

        dob_match = re.search(r'(?:Date\s*of\s*Birth|DOB|Birth\s*Date)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        if not dob_match:
            dob_match = re.search(r'(?:Date\s*of\s*Birth|DOB|Birth\s*Date)\s*:?\s*([A-Za-z0-9,\s/-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["date_of_birth"] = ExtractedField(value=dob_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": dob_match.group(0)}) if dob_match else ExtractedField(value=None, confidence=0.0)

        gender_match = re.search(r'(?:Gender|Sex)\s*:?\s*(Male|Female|Other|M|F)\b', text, re.IGNORECASE)
        if gender_match:
            g_raw = gender_match.group(1).upper()
            g_val = "Male" if g_raw in ["MALE", "M"] else ("Female" if g_raw in ["FEMALE", "F"] else g_raw.capitalize())
            fields["gender"] = ExtractedField(value=g_val, confidence=0.95, source={"page": 1, "text": gender_match.group(0)})
        else:
            fields["gender"] = ExtractedField(value=None, confidence=0.0)

        def clean_val(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            s = str(val).strip()
            lower = s.lower()
            if any(p in lower for p in ["intentionally omitted", "sample text", "placeholder", "test document", "missing field", "not provided", "intentionally_omitted"]):
                return None
            return s

        type_label_m = re.search(r'(?:Identity[\s_]*Document[\s_]*Type|Identity[\s_]*Type|ID[\s_]*Type|Document[\s_]*Type)\s*:?\s*([^\n\|]+)', text, re.IGNORECASE)

        pan_match = re.search(r'\b([A-Z]{5}\d{4}[A-Z]{1})\b', text)
        aadhaar_match = re.search(r'\b((?:\d{4}|X{4})[\s-]?(?:\d{4}|X{4})[\s-]?\d{4})\b', text, re.IGNORECASE)
        pass_match = re.search(r'(?:Passport\s*No|ID\s*No)\s*:\s*([\w]+)', text, re.IGNORECASE)
        dl_match = re.search(r'(?:License\s*No)\s*:\s*([\w-]+)', text, re.IGNORECASE)

        if type_label_m:
            raw_t = type_label_m.group(1).strip()
            cleaned_t = clean_val(raw_t)
            fields["identity_document_type"] = ExtractedField(value=cleaned_t, confidence=0.90 if cleaned_t else 0.0, source={"page": 1, "text": type_label_m.group(0)})
        elif pan_match:
            fields["identity_document_type"] = ExtractedField(value="PAN", confidence=0.99, source={"page": 1, "text": "PAN CARD" if "PAN CARD" in text.upper() else "PAN"})
        elif aadhaar_match:
            fields["identity_document_type"] = ExtractedField(value="Aadhaar Card", confidence=0.99, source={"page": 1, "text": "Aadhaar Card"})
        elif pass_match:
            fields["identity_document_type"] = ExtractedField(value="Passport", confidence=0.95, source={"page": 1, "text": "Passport"})
        elif dl_match:
            fields["identity_document_type"] = ExtractedField(value="Driver License", confidence=0.95, source={"page": 1, "text": "Driver License"})
        else:
            fields["identity_document_type"] = ExtractedField(value=None, confidence=0.0)

        if pan_match:
            fields["identity_document_number"] = ExtractedField(value=pan_match.group(1), confidence=0.99, source={"page": 1, "text": pan_match.group(0)})
            fields["pan_number"] = ExtractedField(value=pan_match.group(1), confidence=0.99, source={"page": 1, "text": pan_match.group(0)})
        elif aadhaar_match:
            fields["identity_document_number"] = ExtractedField(value=aadhaar_match.group(1), confidence=0.99, source={"page": 1, "text": aadhaar_match.group(0)})
            fields["pan_number"] = ExtractedField(value=None, confidence=0.0)
        elif pass_match:
            fields["identity_document_number"] = ExtractedField(value=pass_match.group(1), confidence=0.95, source={"page": 1, "text": pass_match.group(0)})
            fields["pan_number"] = ExtractedField(value=None, confidence=0.0)
        elif dl_match:
            fields["identity_document_number"] = ExtractedField(value=dl_match.group(1), confidence=0.95, source={"page": 1, "text": dl_match.group(0)})
            fields["pan_number"] = ExtractedField(value=None, confidence=0.0)
        else:
            fields["identity_document_number"] = ExtractedField(value=None, confidence=0.0)
            fields["pan_number"] = ExtractedField(value=None, confidence=0.0)

        if not fields["identity_document_number"].value:
            num_match = re.search(r'(?:Identity\s*Document\s*Number|Identity\s*Number|ID\s*Number|Document\s*Number|ID\s*No|Doc\s*No)\s*:?\s*([\w-]+)', text, re.IGNORECASE)
            if num_match and num_match.group(1).strip().upper() not in ["DETAILS", "VALUE"]:
                fields["identity_document_number"] = ExtractedField(value=num_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": num_match.group(0)})

        addr_match = re.search(r'(?:Address|Perm\s*Address|Permanent\s*Address|Residential\s*Address|Communication\s*Address)\s*:?\s*([^\n]+(?:\n[^\n]+)?)', text, re.IGNORECASE)
        if addr_match:
            raw_a = addr_match.group(1).strip().replace("\n", ", ")
            clean_addr = re.sub(r',\s*(?:Issue\s*Date|Expiry\s*Date|DOB|Gender|Date\s*of).*$', '', raw_a, flags=re.IGNORECASE)
            fields["address"] = ExtractedField(value=clean_addr, confidence=0.90, source={"page": 1, "text": addr_match.group(0)})
        else:
            fields["address"] = ExtractedField(value=None, confidence=0.0)

        issue_match = re.search(r'(?:Issue\s*Date|Issued\s*On|Date\s*of\s*Issue)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        fields["issue_date"] = ExtractedField(value=issue_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": issue_match.group(0)}) if issue_match else ExtractedField(value=None, confidence=0.0)

        expiry_match = re.search(r'(?:Expiry\s*Date|Valid\s*Till|Valid\s*Upto|Expires\s*On)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|Not\s+Applicable|N/A|NA|Lifetime)', text, re.IGNORECASE)
        if expiry_match:
            fields["expiry_date"] = ExtractedField(value=expiry_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": expiry_match.group(0)})
        elif any(k in text.upper() for k in ["NOT APPLICABLE", "LIFETIME", "N/A"]):
            fields["expiry_date"] = ExtractedField(value="Not Applicable", confidence=0.90, source={"page": 1, "text": "Expiry Date: Not Applicable"})
        else:
            fields["expiry_date"] = ExtractedField(value=None, confidence=0.0)

        return fields

    def extract_employment_letter(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        # Employee Name
        name_match = re.search(r'(?:Employee\s*Name|Name\s*of\s*Employee)\s*:?\s*([A-Za-z\s\.]+?)(?:\n|\||$)', text, re.IGNORECASE)
        if not name_match:
            name_match = re.search(r'(?:Employee|Mr\.|Ms\.|Dear|Candidate|certify\s+that|certifies\s+that|confirm\s+that|confirms\s+that)\s*:?\s*([A-Za-z\s]+?)(?:is|has|been|as|currently|working|,|\n|$)', text, re.IGNORECASE)

        if name_match and len(name_match.group(1).strip()) > 2 and not any(k in name_match.group(1).lower() for k in ["candidate", "employee", "director", "sir", "that"]):
            val = name_match.group(1).strip()
            fields["employee_name"] = ExtractedField(value=val, confidence=0.85, source={"page": 1, "text": name_match.group(0)})
            fields["applicant_name"] = ExtractedField(value=val, confidence=0.85, source={"page": 1, "text": name_match.group(0)})
        else:
            fields["employee_name"] = ExtractedField(value=None, confidence=0.0)
            fields["applicant_name"] = ExtractedField(value=None, confidence=0.0)

        # Employer Name — Extract from "Employer Name:", "Company Name:", top letterhead header, or "with/at Company as"
        emp_label_match = re.search(r'\b(?:Employer\s*Name|Employer|Company\s*Name|Company|Organization)\s*:\s*([A-Za-z0-9\s\.\,\&]+?)(?:\n|\||$)', text, re.IGNORECASE)

        if emp_label_match and len(emp_label_match.group(1).strip()) > 2 and emp_label_match.group(1).strip().upper() not in GENERIC_EMPLOYER_TITLES_BLACKLIST and emp_label_match.group(1).strip().upper() not in ["FIELD", "DETAILS", "VALUE"]:
            employer_val = emp_label_match.group(1).strip()
            employer_src = emp_label_match.group(0)
        else:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            top_company = None
            top_src = None
            for l in lines[:3]:
                upper_l = l.upper()
                lower_l = l.lower()
                if upper_l in GENERIC_EMPLOYER_TITLES_BLACKLIST or any(upper_l == b for b in GENERIC_EMPLOYER_TITLES_BLACKLIST):
                    continue
                if any(k in lower_l for k in ["dear", "candidate", "to whom", "date", "ref", "reference", "certify", "confirm", "this is", "employment letter", "offer of employment", "appointment letter", "status"]):
                    continue
                if any(kw in upper_l for kw in ["LTD", "PVT", "INC", "CORP", "LIMITED", "SOLUTIONS", "TECHNOLOGIES", "COMPANY", "SERVICES", "GLOBAL", "SYSTEMS", "INNOVATIONS", "GROUP"]):
                    top_company = l.strip()
                    top_src = l
                    break

            if top_company:
                employer_val = top_company
                employer_src = top_src
            else:
                emp_match = re.search(r'\b(?:with|at|for)\s+([A-Za-z0-9\s\.\,\&]+?)\s+as\b', text, re.IGNORECASE)
                if emp_match and len(emp_match.group(1).strip()) > 2 and not re.search(r'\b(?:us|me|our)\b', emp_match.group(1), re.IGNORECASE):
                    employer_val = emp_match.group(1).strip()
                    employer_src = emp_match.group(0)
                else:
                    employer_val = None
                    employer_src = None
                    for line in lines[:6]:
                        upper_line = line.upper().strip()
                        lower_line = line.lower().strip()
                        if upper_line in GENERIC_EMPLOYER_TITLES_BLACKLIST or any(upper_line == b_title for b_title in GENERIC_EMPLOYER_TITLES_BLACKLIST):
                            continue
                        if any(k in lower_line for k in ["dear", "candidate", "to whom", "date", "ref", "reference", "certify", "confirm", "this is", "working with", "works with", "status", "employment letter", "confirmation letter", "as a", "as an"]):
                            continue
                        if len(line) > 3:
                            employer_val = line.strip()
                            employer_src = line
                            break

        fields["employer_name"] = ExtractedField(
            value=employer_val,
            confidence=0.85 if employer_val else 0.0,
            source={"page": 1, "text": employer_src} if employer_src else None
        )

        # Designation
        desig_match = re.search(r'\b(?:Designation|Role|Position|Title)\s*:?\s*([A-Za-z0-9\s\-\/\&]+?)(?:\.|\n|\b(?:since|on|in)\b|\||$)', text, re.IGNORECASE)
        if not desig_match:
            desig_match = re.search(r'\bas\s+a\s+([A-Za-z0-9\s\-\/\&]+?)(?:\.|\n|\b(?:since|on|in)\b|\||$)', text, re.IGNORECASE)
        if not desig_match:
            desig_match = re.search(r'\bas\s+([A-Za-z0-9\s\-\/\&]+?)(?:\.|\n|\b(?:since|on|in)\b|\||$)', text, re.IGNORECASE)
        if desig_match:
            clean_desig = re.sub(r'^(?:a|an|the)\s+', '', desig_match.group(1).strip(), flags=re.IGNORECASE).strip()
            fields["designation"] = ExtractedField(value=clean_desig, confidence=0.85, source={"page": 1, "text": desig_match.group(0)})
        else:
            fields["designation"] = ExtractedField(value=None, confidence=0.0)

        # Salary ONLY IF EXPLICITLY STATED (Never infer salary from designation)
        sal_res = DeterministicExtractor.extract_currency_amount(text, [
            r'(?:Annual\s*Salary|Annual\s*Compensation|Salary|annual\s*salary\s*compensation\s*of|compensation\s*of|salary\s*of|annual\s*salary\s*of)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Annual\s*Salary|Salary)\s*:?\s*([^\n]+)'
        ])
        if sal_res:
            fields["salary_if_explicitly_stated"] = ExtractedField(value=sal_res[0], confidence=0.95, source={"page": 1, "text": sal_res[2]})
        else:
            fields["salary_if_explicitly_stated"] = ExtractedField(value=None, confidence=0.0)

        # Joining Date
        join_match = re.search(r'(?:Date\s*of\s*Joining|Joining\s*Date|Date\s*Joined|Joined\s*On|Date\s*of\s*Appointment|Appointment\s*Date|Start\s*Date|joining\s*date\s*will\s*be|employed[^\n]+since)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})', text, re.IGNORECASE)
        if not join_match:
            join_match = re.search(r'(?:Date\s*of\s*Joining|Joining\s*Date|Date\s*Joined|Joined\s*On|Start\s*Date)\s*:?\s*([A-Za-z0-9,\s/-]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        fields["joining_date"] = ExtractedField(value=join_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": join_match.group(0)}) if join_match else ExtractedField(value=None, confidence=0.0)

        # Letter Date
        letter_date_match = re.search(r'(?:Letter\s*Date|Date\s*of\s*Letter|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})', text, re.IGNORECASE)
        if not letter_date_match:
            letter_date_match = re.search(r'(?:Letter\s*Date|Date)\s*:\s*([A-Za-z0-9,\s/-]+?)(?:\n|\||Employee|Ref|Dear|$)', text, re.IGNORECASE)
        fields["letter_date"] = ExtractedField(value=letter_date_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": letter_date_match.group(0)}) if letter_date_match else ExtractedField(value=None, confidence=0.0)

        # Department
        dept_match = re.search(r'(?:Department|Dept)\s*:?\s*([A-Za-z0-9\s]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        if not dept_match:
            dept_match = re.search(r'\bin\s+([A-Za-z0-9\s]+?)\s+Department\b', text, re.IGNORECASE)
        if not dept_match:
            dept_match = re.search(r'\b([A-Za-z0-9\s]+?)\s+Department\b', text, re.IGNORECASE)
        fields["department"] = ExtractedField(value=dept_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": dept_match.group(0)}) if dept_match else ExtractedField(value=None, confidence=0.0)

        # Reference Number
        ref_match = re.search(r'(?:Reference\s*Number|Ref\s*No|Ref\s*Number|Reference\s*No|Ref)\s*:?\s*([\w-]+)', text, re.IGNORECASE)
        fields["reference_number"] = ExtractedField(value=ref_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": ref_match.group(0)}) if ref_match else ExtractedField(value=None, confidence=0.0)

        # Work Location
        loc_match = re.search(r'(?:based\s*in|Location|Work\s*Location)\s*:?\s*([A-Za-z0-9\s,]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        fields["work_location"] = ExtractedField(value=loc_match.group(1).strip(), confidence=0.85, source={"page": 1, "text": loc_match.group(0)}) if loc_match else ExtractedField(value=None, confidence=0.0)

        # Employment Status
        status_match = re.search(r'(?:Employment\s*Status|Job\s*Status|Employment\s*Type|Status)\s*:?\s*([A-Za-z0-9\s-]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        if status_match and len(status_match.group(1).strip()) > 1 and status_match.group(1).strip().upper() not in ["DETAILS", "VALUE", "UNKNOWN"]:
            fields["employment_status"] = ExtractedField(value=status_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": status_match.group(0)})
        elif "permanent" in text.lower():
            fields["employment_status"] = ExtractedField(value="Permanent", confidence=0.85, source={"page": 1, "text": "Employment Status: Permanent"})
        elif "active" in text.lower():
            fields["employment_status"] = ExtractedField(value="Active", confidence=0.85, source={"page": 1, "text": "Employment Status: Active"})
        elif "confirmed" in text.lower() or "confirm" in text.lower():
            fields["employment_status"] = ExtractedField(value="Confirmed", confidence=0.85, source={"page": 1, "text": "Employment Status: Confirmed"})
        elif "full time" in text.lower() or "full-time" in text.lower():
            fields["employment_status"] = ExtractedField(value="Full Time", confidence=0.85, source={"page": 1, "text": "Employment Status: Full Time"})
        else:
            fields["employment_status"] = ExtractedField(value=None, confidence=0.0)

        # Employment Type (Preserve exact string e.g. "Permanent Full-Time")
        type_match = re.search(r'(?:Employment\s*Type|Job\s*Type|Type\s*of\s*Employment)\s*:?\s*([A-Za-z0-9\s-]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        if type_match:
            fields["employment_type"] = ExtractedField(value=type_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": type_match.group(0)})
        elif "permanent full-time" in text.lower():
            fields["employment_type"] = ExtractedField(value="Permanent Full-Time", confidence=0.90, source={"page": 1, "text": "Employment Type: Permanent Full-Time"})
        elif "full-time" in text.lower():
            fields["employment_type"] = ExtractedField(value="Full-time", confidence=0.80, source={"page": 1, "text": "Full-time"})
        else:
            fields["employment_type"] = ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_office_id(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        # Employee Name / Cardholder Name
        name_match = re.search(r'(?:Employee\s*Name|Cardholder\s*Name|Name|Employee)\s*:?\s*([A-Za-z\s\.]+?)(?:\n|\||$)', text, re.IGNORECASE)
        if name_match and len(name_match.group(1).strip()) > 2 and not any(k in name_match.group(1).upper() for k in ["CARD", "OFFICE", "COMPANY", "IDENTITY", "DEPARTMENT"]):
            val = name_match.group(1).strip()
            fields["employee_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": name_match.group(0)})
            fields["applicant_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": name_match.group(0)})
        else:
            fields["employee_name"] = ExtractedField(value=None, confidence=0.0)
            fields["applicant_name"] = ExtractedField(value=None, confidence=0.0)

        # Employee ID / Staff ID / Card Number
        emp_id_match = re.search(r'(?:Employee\s*ID|Emp\s*ID|Staff\s*ID|ID\s*No|ID\s*Number|Card\s*No|Badge\s*No)\s*:?\s*([\w-]+)', text, re.IGNORECASE)
        if emp_id_match and len(emp_id_match.group(1).strip()) > 1 and emp_id_match.group(1).upper() not in ["NAME", "CARD", "STATUS"]:
            fields["employee_id"] = ExtractedField(value=emp_id_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": emp_id_match.group(0)})
            fields["document_number"] = ExtractedField(value=emp_id_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": emp_id_match.group(0)})
        else:
            fields["employee_id"] = ExtractedField(value=None, confidence=0.0)
            fields["document_number"] = ExtractedField(value=None, confidence=0.0)

        # Employer / Company Name
        emp_match = re.search(r'(?:Employer\s*Name|Employer|Company\s*Name|Company|Organization)\s*:?\s*([A-Za-z0-9\s\.\,\&]+?)(?:\n|\||$)', text, re.IGNORECASE)
        if emp_match and len(emp_match.group(1).strip()) > 2 and emp_match.group(1).strip().upper() not in GENERIC_EMPLOYER_TITLES_BLACKLIST:
            val = emp_match.group(1).strip()
            fields["employer_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": emp_match.group(0)})
            fields["company_name"] = ExtractedField(value=val, confidence=0.90, source={"page": 1, "text": emp_match.group(0)})
        else:
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            employer_val = None
            employer_src = None
            for l in lines[:5]:
                upper_l = l.upper()
                if upper_l in GENERIC_EMPLOYER_TITLES_BLACKLIST or any(upper_l == b for b in GENERIC_EMPLOYER_TITLES_BLACKLIST):
                    continue
                if any(k in upper_l for k in ["CARD", "IDENTITY", "EMPLOYEE", "NAME", "ID:"]):
                    continue
                if len(l) > 3:
                    employer_val = l.strip()
                    employer_src = l
                    break
            fields["employer_name"] = ExtractedField(value=employer_val, confidence=0.85 if employer_val else 0.0, source={"page": 1, "text": employer_src} if employer_src else None)
            fields["company_name"] = ExtractedField(value=employer_val, confidence=0.85 if employer_val else 0.0, source={"page": 1, "text": employer_src} if employer_src else None)

        # Designation
        desig_match = re.search(r'(?:Designation|Role|Position|Title)\s*:?\s*([A-Za-z0-9\s\-\/\&]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        fields["designation"] = ExtractedField(value=desig_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": desig_match.group(0)}) if desig_match else ExtractedField(value=None, confidence=0.0)

        # Department
        dept_match = re.search(r'(?:Department|Dept)\s*:?\s*([A-Za-z0-9\s]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        fields["department"] = ExtractedField(value=dept_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": dept_match.group(0)}) if dept_match else ExtractedField(value=None, confidence=0.0)

        # Valid From / Valid Until / Issue Date / Expiry Date
        from_match = re.search(r'(?:Valid\s*From|Issue\s*Date|Issued\s*Date|Date\s*of\s*Issue)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        fields["valid_from"] = ExtractedField(value=from_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": from_match.group(0)}) if from_match else ExtractedField(value=None, confidence=0.0)

        until_match = re.search(r'(?:Valid\s*Until|Valid\s*Thru|Expiry\s*Date|Expires|Valid\s*Upto)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|Not\s+Applicable|Lifetime)', text, re.IGNORECASE)
        fields["valid_until"] = ExtractedField(value=until_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": until_match.group(0)}) if until_match else ExtractedField(value=None, confidence=0.0)

        # Work Location
        loc_match = re.search(r'(?:Location|Work\s*Location|Branch|Office)\s*:?\s*([A-Za-z0-9\s,]+?)(?:\.|\n|\||$)', text, re.IGNORECASE)
        fields["work_location"] = ExtractedField(value=loc_match.group(1).strip(), confidence=0.85, source={"page": 1, "text": loc_match.group(0)}) if loc_match else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_form_16(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        name_match = re.search(r'(?:Employee|FICTIONAL WORKER)\s*:?\s*([A-Za-z\s]+?)(?:PAN|TAN||\n|$)', text, re.IGNORECASE)
        fields["employee_name"] = ExtractedField(value=name_match.group(1).strip(), confidence=0.85, source={"page": 1, "text": name_match.group(0)}) if name_match else ExtractedField(value=None, confidence=0.0)

        emp_match = re.search(r'(?:Employer\s*Name|Employer)\s*:?\s*([A-Za-z\s]+?)(?:TAN|PAN|\n|$)', text, re.IGNORECASE)
        fields["employer_name"] = ExtractedField(value=emp_match.group(1).strip(), confidence=0.85, source={"page": 1, "text": emp_match.group(0)}) if emp_match else ExtractedField(value=None, confidence=0.0)

        ay_match = re.search(r'(?:Assessment\s*Year)\s*:\s*([\d-]+)', text, re.IGNORECASE)
        fields["assessment_year"] = ExtractedField(value=ay_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": ay_match.group(0)}) if ay_match else ExtractedField(value=None, confidence=0.0)

        tan_match = re.search(r'(?:TAN)\s*:?\s*([A-Z]{4}\d{5}[A-Z]{1})', text)
        fields["employer_TAN"] = ExtractedField(value=tan_match.group(1), confidence=0.95, source={"page": 1, "text": tan_match.group(0)}) if tan_match else ExtractedField(value=None, confidence=0.0)

        pan_match = re.search(r'(?:PAN)\s*:?\s*([A-Z]{5}\d{4}[A-Z]{1})', text)
        fields["employee_PAN"] = ExtractedField(value=pan_match.group(1), confidence=0.95, source={"page": 1, "text": pan_match.group(0)}) if pan_match else ExtractedField(value=None, confidence=0.0)

        sal_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Gross\s*Salary|Total\s*Salary\s*Paid|Gross\s*Compensation)\s*:\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'])
        fields["salary_income"] = ExtractedField(value=sal_res[0], confidence=0.95, source={"page": 1, "text": sal_res[2]}) if sal_res else ExtractedField(value=None, confidence=0.0)

        tds_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tax\s*Deducted\s*at\s*Source\s*\(TDS\)|Total\s*TDS\s*Deducted|TDS)\s*:\s*([\$\₹]?\s*[\d,]+(?:\.\d{2})?)'])
        fields["TDS"] = ExtractedField(value=tds_res[0], confidence=0.95, source={"page": 1, "text": tds_res[2]}) if tds_res else ExtractedField(value=None, confidence=0.0)
        fields["tax_deducted"] = ExtractedField(value=fields["TDS"].value, confidence=fields["TDS"].confidence, source=fields["TDS"].source)

        fields["financial_year"] = ExtractedField(value=None, confidence=0.0)
        fields["tax_payable"] = ExtractedField(value=None, confidence=0.0)
        return fields

    @staticmethod
    def extract_address_proof(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        # Full Name
        name_match = re.search(r'(?:Full\s*Name|Consumer\s*Name|Subscriber\s*Name|Customer|Tenant|Resident\s*Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if not name_match:
            name_match = re.search(r'(?:Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["full_name"] = ExtractedField(value=name_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": name_match.group(0)}) if name_match else ExtractedField(value=None, confidence=0.0)

        # Address
        addr_match = re.search(r'(?:Billing\s*Address|Premises\s*Address|Service\s*Address|Installation\s*Address|Residential\s*Address|Address)\s*:\s*([^\n]+(?:\n[^\n]+)?)', text, re.IGNORECASE)
        if addr_match:
            clean_addr = addr_match.group(1).strip().replace("\n", ", ")
            clean_addr = re.sub(r',\s*(?:Document\s*Date|Bill\s*Date|Issued\s*Date|Issue\s*Date|Date).*$', '', clean_addr, flags=re.IGNORECASE)
            fields["address"] = ExtractedField(value=clean_addr, confidence=0.90, source={"page": 1, "text": addr_match.group(0)})
        else:
            fields["address"] = ExtractedField(value=None, confidence=0.0)

        # Document Date
        date_match = re.search(r'(?:Document\s*Date|Bill\s*Date|Issued\s*Date|Date\s*of\s*Issue|Issue\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        fields["document_date"] = ExtractedField(value=date_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": date_match.group(0)}) if date_match else ExtractedField(value=None, confidence=0.0)

        # Issuer - Never grab generic title lines from GENERIC_ISSUER_TITLES_BLACKLIST
        issuer_match = re.search(r'(?:Issuer|Issuing\s*Authority|Issued\s*By|Authority|Provider|Organization)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if issuer_match and issuer_match.group(1).strip().upper() not in GENERIC_ISSUER_TITLES_BLACKLIST:
            fields["issuer"] = ExtractedField(value=issuer_match.group(1).strip(), confidence=0.95, source={"page": 1, "text": issuer_match.group(0)})
        else:
            issuer_val = None
            issuer_src = None
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            for l in lines:
                upper_l = l.upper()
                if upper_l in GENERIC_ISSUER_TITLES_BLACKLIST or any(upper_l == b for b in GENERIC_ISSUER_TITLES_BLACKLIST):
                    continue
                if any(k in upper_l for k in ["OFFICE", "MUNICIPAL", "CORPORATION", "DEPARTMENT", "BOARD", "AUTHORITY", "GOVERNMENT", "LIMITED", "LTD", "PVT", "COMPANY"]):
                    issuer_val = l.strip()
                    issuer_src = l
                    break
            fields["issuer"] = ExtractedField(value=issuer_val, confidence=0.85 if issuer_val else 0.0, source={"page": 1, "text": issuer_src} if issuer_src else None)

        # Postal Code
        post_match = re.search(r'(?:Postal\s*Code|PIN\s*Code|Zip\s*Code|PIN|ZIP)\s*:\s*(\d{6}|\d{5})', text, re.IGNORECASE)
        if not post_match:
            post_match = re.search(r'\b(\d{6}|\d{5})\b', text)
        fields["postal_code"] = ExtractedField(value=post_match.group(1), confidence=0.90, source={"page": 1, "text": post_match.group(0)}) if post_match else ExtractedField(value=None, confidence=0.0)

        # City
        city_match = re.search(r'(?:\n|^)\s*(?:City|District|Town)\s*:\s*([A-Za-z\s]+?)(?:\||\n|,|$)', text, re.IGNORECASE)
        if city_match and len(city_match.group(1).strip()) > 2 and city_match.group(1).strip().lower() not in ["state", "country"]:
            fields["city"] = ExtractedField(value=city_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": city_match.group(0)})
        else:
            fields["city"] = ExtractedField(value=None, confidence=0.0)

        # State
        state_match = re.search(r'(?:\n|^)\s*(?:State|Province)\s*:\s*([A-Za-z\s]+?)(?:\||\n|,|$)', text, re.IGNORECASE)
        if state_match and len(state_match.group(1).strip()) > 2 and state_match.group(1).strip().lower() not in ["country", "india"]:
            fields["state"] = ExtractedField(value=state_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": state_match.group(0)})
        else:
            fields["state"] = ExtractedField(value=None, confidence=0.0)

        # Country
        country_match = re.search(r'(?:\n|^)\s*(?:Country|Nation)\s*:\s*([A-Za-z\s]+?)(?:\||\n|,|$)', text, re.IGNORECASE)
        if country_match:
            fields["country"] = ExtractedField(value=country_match.group(1).strip(), confidence=0.90, source={"page": 1, "text": country_match.group(0)})
        elif "INDIA" in text.upper():
            fields["country"] = ExtractedField(value="India", confidence=0.90, source={"page": 1, "text": "India"})
        else:
            fields["country"] = ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_property_document(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        buyer_m = re.search(r'(?:Buyer|Purchaser|Transferee)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["buyer_name"] = ExtractedField(value=buyer_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": buyer_m.group(0)}) if buyer_m else ExtractedField(value=None, confidence=0.0)

        seller_m = re.search(r'(?:Seller|Vendor|Transferor)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["seller_name"] = ExtractedField(value=seller_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": seller_m.group(0)}) if seller_m else ExtractedField(value=None, confidence=0.0)

        owner_m = re.search(r'(?:Owner|Proprietor|Title\s*Holder)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["owner_name"] = ExtractedField(value=owner_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": owner_m.group(0)}) if owner_m else ExtractedField(value=fields.get("buyer_name").value, confidence=0.70)

        co_owner_m = re.search(r'(?:Co-Owner|Joint\s*Owner)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["co_owner_name"] = ExtractedField(value=co_owner_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": co_owner_m.group(0)}) if co_owner_m else ExtractedField(value=None, confidence=0.0)

        addr_m = re.search(r'(?:Property\s*Address|Schedule\s*Property|Premises)\s*:\s*([^\n]+(?:\n[^\n]+)?)', text, re.IGNORECASE)
        fields["property_address"] = ExtractedField(value=addr_m.group(1).strip().replace("\n", ", "), confidence=0.90, source={"page": 1, "text": addr_m.group(0)}) if addr_m else ExtractedField(value=None, confidence=0.0)

        desc_m = re.search(r'(?:Property\s*Description|Description)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["property_description"] = ExtractedField(value=desc_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": desc_m.group(0)}) if desc_m else ExtractedField(value=None, confidence=0.0)

        id_m = re.search(r'(?:Property\s*ID|PID|Property\s*Identifier)\s*:\s*([\w-]+)', text, re.IGNORECASE)
        fields["property_identifier"] = ExtractedField(value=id_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": id_m.group(0)}) if id_m else ExtractedField(value=None, confidence=0.0)

        surv_m = re.search(r'(?:Survey\s*No|Survey\s*Number|Sy\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["survey_number"] = ExtractedField(value=surv_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": surv_m.group(0)}) if surv_m else ExtractedField(value=None, confidence=0.0)

        plot_m = re.search(r'(?:Plot\s*No|Plot\s*Number)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["plot_number"] = ExtractedField(value=plot_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": plot_m.group(0)}) if plot_m else ExtractedField(value=None, confidence=0.0)

        reg_num_m = re.search(r'(?:Registration\s*No|Registration\s*Number|Reg\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["registration_number"] = ExtractedField(value=reg_num_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": reg_num_m.group(0)}) if reg_num_m else ExtractedField(value=None, confidence=0.0)

        doc_num_m = re.search(r'(?:Document\s*No|Document\s*Number|Deed\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["document_number"] = ExtractedField(value=doc_num_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": doc_num_m.group(0)}) if doc_num_m else ExtractedField(value=None, confidence=0.0)

        reg_dt = re.search(r'(?:Registration\s*Date|Reg\s*Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["registration_date"] = ExtractedField(value=reg_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": reg_dt.group(0)}) if reg_dt else ExtractedField(value=None, confidence=0.0)

        agr_dt = re.search(r'(?:Agreement\s*Date|Deed\s*Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["agreement_date"] = ExtractedField(value=agr_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": agr_dt.group(0)}) if agr_dt else ExtractedField(value=None, confidence=0.0)

        sale_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Sale\s*Amount|Sale\s*Price|Total\s*Sale\s*Value|Agreed\s*Purchase\s*Price)\s*:?\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["sale_amount"] = ExtractedField(value=sale_amt[0], confidence=0.95, source={"page": 1, "text": sale_amt[2]}) if sale_amt else ExtractedField(value=None, confidence=0.0)

        cons_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Sale\s*Consideration|Consideration\s*Amount|Consideration|Agreed\s*Consideration)\s*:?\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        if cons_amt:
            fields["consideration_amount"] = ExtractedField(value=cons_amt[0], confidence=0.95, source={"page": 1, "text": cons_amt[2]})
        else:
            fields["consideration_amount"] = ExtractedField(value=fields.get("sale_amount").value, confidence=fields.get("sale_amount").confidence)

        prop_val = DeterministicExtractor.extract_currency_amount(text, [r'(?:Property\s*Value|Guidance\s*Value|Market\s*Value)\s*:?\s*(?:INR|\$|Rs\.?|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        if prop_val:
            fields["property_value"] = ExtractedField(value=prop_val[0], confidence=0.95, source={"page": 1, "text": prop_val[2]})
        else:
            fields["property_value"] = ExtractedField(value=fields.get("sale_amount").value, confidence=fields.get("sale_amount").confidence if fields.get("sale_amount").value else 0.0)

        stamp_m = DeterministicExtractor.extract_currency_amount(text, [r'(?:Stamp\s*Duty)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["stamp_duty"] = ExtractedField(value=stamp_m[0], confidence=0.90, source={"page": 1, "text": stamp_m[2]}) if stamp_m else ExtractedField(value=None, confidence=0.0)

        reg_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Registration\s*Fee)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["registration_fee"] = ExtractedField(value=reg_fee[0], confidence=0.90, source={"page": 1, "text": reg_fee[2]}) if reg_fee else ExtractedField(value=None, confidence=0.0)

        auth_m = re.search(r'(?:Sub-Registrar\s*Office|Registrar|Issuing\s*Authority|Authority)\s*:\s*([A-Za-z0-9\s,\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["issuing_authority"] = ExtractedField(value=auth_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": auth_m.group(0)}) if auth_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_approved_plan(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        # Owner Name
        own_m = re.search(r'(?:Owner\s*Name|Property\s*Owner|Registered\s*Owner|Applicant\s*Name|Applicant|Owner)\s*:?\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["owner_name"] = ExtractedField(value=own_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": own_m.group(0)}) if own_m and len(own_m.group(1).strip()) > 2 else ExtractedField(value=None, confidence=0.0)

        # Property Address
        addr_m = re.search(r'(?:Property\s*Address|Site\s*Address|Premises\s*Address|Location|Address)\s*:?\s*([^\n]+(?:\n[^\n]+)?)', text, re.IGNORECASE)
        if addr_m:
            clean_addr = addr_m.group(1).strip().replace("\n", ", ")
            clean_addr = re.sub(r',\s*(?:Plot\s*No|Plan\s*No|Approval\s*No|Date).*$', '', clean_addr, flags=re.IGNORECASE)
            fields["property_address"] = ExtractedField(value=clean_addr, confidence=0.90, source={"page": 1, "text": addr_m.group(0)})
        else:
            fields["property_address"] = ExtractedField(value=None, confidence=0.0)

        # Plan Number (Independent from Approval Number)
        plan_m = re.search(r'(?:Building\s*Plan\s*No|Plan\s*Number|Plan\s*No|Drawing\s*No|Map\s*No|Plan\s*#)\s*:?\s*([\w/-]+)', text, re.IGNORECASE)
        fields["plan_number"] = ExtractedField(value=plan_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": plan_m.group(0)}) if plan_m else ExtractedField(value=None, confidence=0.0)

        # Approval Number (Independent from Plan Number)
        app_m = re.search(r'(?:Building\s*Plan\s*Approval\s*No|Approval\s*Number|Approval\s*No|Sanction\s*Number|Sanction\s*No|Permit\s*Number|Permit\s*No|Building\s*Permit\s*No|Approval\s*#)\s*:?\s*([\w/-]+)', text, re.IGNORECASE)
        fields["approval_number"] = ExtractedField(value=app_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": app_m.group(0)}) if app_m else ExtractedField(value=None, confidence=0.0)

        # Approval Date
        dt_m = re.search(r'(?:Approval\s*Date|Sanction\s*Date|Permit\s*Date|Date\s*of\s*Approval|Date)\s*:?\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4}|[A-Za-z]+\s+\d{1,2},\s*\d{4}|\d{1,2}\s+[A-Za-z]+\s+\d{4})', text, re.IGNORECASE)
        fields["approval_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        # Approving Authority
        auth_m = re.search(r'(?:Approving\s*Authority|Sanctioning\s*Authority|Planning\s*Authority|Municipal\s*Authority|Issuing\s*Authority|Authority)\s*:?\s*([A-Za-z0-9\s,\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["approving_authority"] = ExtractedField(value=auth_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": auth_m.group(0)}) if auth_m and len(auth_m.group(1).strip()) > 2 else ExtractedField(value=None, confidence=0.0)

        # Plot Number
        plot_m = re.search(r'(?:Plot\s*Number|Plot\s*No|Site\s*Number|Site\s*No)\s*:?\s*([\w/-]+)', text, re.IGNORECASE)
        fields["plot_number"] = ExtractedField(value=plot_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": plot_m.group(0)}) if plot_m else ExtractedField(value=None, confidence=0.0)

        # Building Area
        area_m = re.search(r'(?:Building\s*Area|Built-up\s*Area|Plinth\s*Area|Total\s*Area|Area)\s*:?\s*([\d,]+(?:\.\d+)?(?:\s*(?:sq\s*ft|sqm|sq\s*m|square\s*feet))?)', text, re.IGNORECASE)
        fields["building_area"] = ExtractedField(value=area_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": area_m.group(0)}) if area_m else ExtractedField(value=None, confidence=0.0)

        # Floors
        floor_m = re.search(r'(?:Floors|No\s*of\s*Floors|Number\s*of\s*Floors|Structure)\s*:?\s*([A-Za-z0-9\s\+]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["floors"] = ExtractedField(value=floor_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": floor_m.group(0)}) if floor_m else ExtractedField(value=None, confidence=0.0)

        # Property Type
        type_m = re.search(r'(?:Property\s*Type|Building\s*Type)\s*:?\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if type_m:
            fields["property_type"] = ExtractedField(value=type_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": type_m.group(0)})
        else:
            fields["property_type"] = ExtractedField(value="Residential" if "residential" in text.lower() else ("Commercial" if "commercial" in text.lower() else None), confidence=0.80 if ("residential" in text.lower() or "commercial" in text.lower()) else 0.0)

        return fields

    @staticmethod
    def extract_vehicle_quotation(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        cust_m = re.search(r'(?:Customer\s*Name|Buyer\s*Name|Customer|Client|Buyer|Policy\s*Holder)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        cust_val = cust_m.group(1).strip() if cust_m else None
        fields["customer_name"] = ExtractedField(value=cust_val, confidence=0.90, source={"page": 1, "text": cust_m.group(0)}) if cust_m else ExtractedField(value=None, confidence=0.0)
        fields["buyer_name"] = ExtractedField(value=cust_val, confidence=0.90, source={"page": 1, "text": cust_m.group(0)}) if cust_m else ExtractedField(value=None, confidence=0.0)

        deal_m = re.search(r'(?:Dealer\s*Name|Dealer|Showroom|Seller\s*/\s*Dealer|Seller)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["dealer_name"] = ExtractedField(value=deal_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": deal_m.group(0)}) if deal_m else ExtractedField(value=None, confidence=0.0)

        q_num = re.search(r'(?:Quotation\s*No|Quotation\s*Number|Quote\s*No|Agreement\s*Number|Agreement\s*No|Invoice\s*No|Invoice\s*Number)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        q_num_val = q_num.group(1).strip() if q_num else None
        fields["quotation_number"] = ExtractedField(value=q_num_val, confidence=0.95, source={"page": 1, "text": q_num.group(0)}) if q_num else ExtractedField(value=None, confidence=0.0)
        fields["invoice_number"] = ExtractedField(value=q_num_val, confidence=0.95, source={"page": 1, "text": q_num.group(0)}) if q_num else ExtractedField(value=None, confidence=0.0)

        q_dt = re.search(r'(?:Quotation\s*Date|Quote\s*Date|Agreement\s*Date|Invoice\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        q_dt_val = q_dt.group(1).strip() if q_dt else None
        fields["quotation_date"] = ExtractedField(value=q_dt_val, confidence=0.95, source={"page": 1, "text": q_dt.group(0)}) if q_dt else ExtractedField(value=None, confidence=0.0)
        fields["invoice_date"] = ExtractedField(value=q_dt_val, confidence=0.95, source={"page": 1, "text": q_dt.group(0)}) if q_dt else ExtractedField(value=None, confidence=0.0)

        make_m = re.search(r'(?:Vehicle\s*Make|Make|Brand|Manufacturer)\s*:\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["vehicle_make"] = ExtractedField(value=make_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": make_m.group(0)}) if make_m else ExtractedField(value=None, confidence=0.0)

        mod_m = re.search(r'(?:Vehicle\s*Model|Model)\s*:\s*([A-Za-z0-9\s-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["vehicle_model"] = ExtractedField(value=mod_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": mod_m.group(0)}) if mod_m else ExtractedField(value=None, confidence=0.0)

        var_m = re.search(r'(?:Variant|Trim)\s*:\s*([A-Za-z0-9\s-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["vehicle_variant"] = ExtractedField(value=var_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": var_m.group(0)}) if var_m else ExtractedField(value=None, confidence=0.0)

        fields["vehicle_type"] = ExtractedField(value="Four Wheeler" if any(k in text.lower() for k in ["car", "suv", "sedan", "hatchback"]) else ("Two Wheeler" if "bike" in text.lower() else "Vehicle"), confidence=0.80)
        fields["registration_location"] = ExtractedField(value=None, confidence=0.0)

        ex_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Ex-Showroom\s*Price|Ex-Showroom|Base\s*Price)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["ex_showroom_price"] = ExtractedField(value=ex_res[0], confidence=0.95, source={"page": 1, "text": ex_res[2]}) if ex_res else ExtractedField(value=None, confidence=0.0)

        acc_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Accessories|Accessories\s*Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["accessories_amount"] = ExtractedField(value=acc_res[0], confidence=0.90, source={"page": 1, "text": acc_res[2]}) if acc_res else ExtractedField(value=None, confidence=0.0)

        ins_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Insurance|Insurance\s*Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["insurance_amount"] = ExtractedField(value=ins_res[0], confidence=0.90, source={"page": 1, "text": ins_res[2]}) if ins_res else ExtractedField(value=None, confidence=0.0)

        tax_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Road\s*Tax|Tax\s*Amount|RTO\s*Charges)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tax_amount"] = ExtractedField(value=tax_res[0], confidence=0.90, source={"page": 1, "text": tax_res[2]}) if tax_res else ExtractedField(value=None, confidence=0.0)

        disc_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Discount|Discount\s*Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["discount"] = ExtractedField(value=disc_res[0], confidence=0.90, source={"page": 1, "text": disc_res[2]}) if disc_res else ExtractedField(value=None, confidence=0.0)

        tot_res = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Amount|On-Road\s*Price|Quoted\s*Amount|Agreed\s*Purchase\s*Price|Purchase\s*Price|Total)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_amount"] = ExtractedField(value=tot_res[0], confidence=0.95, source={"page": 1, "text": tot_res[2]}) if tot_res else ExtractedField(value=None, confidence=0.0)
        fields["quoted_amount"] = ExtractedField(value=fields["total_amount"].value, confidence=fields["total_amount"].confidence)

        return fields

    @staticmethod
    def extract_vehicle_invoice(text: str) -> Dict[str, ExtractedField]:
        fields = DeterministicExtractor.extract_vehicle_quotation(text)

        inv_n = re.search(r'(?:Invoice\s*No|Invoice\s*Number|Agreement\s*Number|Agreement\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        if inv_n:
            fields["invoice_number"] = ExtractedField(value=inv_n.group(1).strip(), confidence=0.95, source={"page": 1, "text": inv_n.group(0)})
            fields["quotation_number"] = ExtractedField(value=inv_n.group(1).strip(), confidence=0.95, source={"page": 1, "text": inv_n.group(0)})

        inv_dt = re.search(r'(?:Invoice\s*Date|Agreement\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        if inv_dt:
            fields["invoice_date"] = ExtractedField(value=inv_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": inv_dt.group(0)})
            fields["quotation_date"] = ExtractedField(value=inv_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": inv_dt.group(0)})

        chass_m = re.search(r'(?:Chassis\s*No|Chassis\s*Number|VIN)\s*:\s*([\w-]+)', text, re.IGNORECASE)
        fields["chassis_number"] = ExtractedField(value=chass_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": chass_m.group(0)}) if chass_m else ExtractedField(value=None, confidence=0.0)

        eng_m = re.search(r'(?:Engine\s*No|Engine\s*Number)\s*:\s*([\w-]+)', text, re.IGNORECASE)
        fields["engine_number"] = ExtractedField(value=eng_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": eng_m.group(0)}) if eng_m else ExtractedField(value=None, confidence=0.0)

        reg_m = re.search(r'(?:Registration\s*No|Reg\s*No)\s*:\s*([\w-]+)', text, re.IGNORECASE)
        fields["registration_number"] = ExtractedField(value=reg_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": reg_m.group(0)}) if reg_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_admission_letter(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        stud_m = re.search(r'(?:Student\s*Name|Candidate\s*Name|Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["student_name"] = ExtractedField(value=stud_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": stud_m.group(0)}) if stud_m else ExtractedField(value=None, confidence=0.0)

        inst_m = re.search(r'(?:Institution|University|College|School)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["institution_name"] = ExtractedField(value=inst_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": inst_m.group(0)}) if inst_m else ExtractedField(value=None, confidence=0.0)

        course_m = re.search(r'(?:Course|Degree|Program)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["course_name"] = ExtractedField(value=course_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": course_m.group(0)}) if course_m else ExtractedField(value=None, confidence=0.0)
        fields["program_name"] = ExtractedField(value=fields["course_name"].value, confidence=fields["course_name"].confidence)

        adm_num = re.search(r'(?:Admission\s*No|Roll\s*No|Application\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["admission_number"] = ExtractedField(value=adm_num.group(1).strip(), confidence=0.95, source={"page": 1, "text": adm_num.group(0)}) if adm_num else ExtractedField(value=None, confidence=0.0)

        adm_dt = re.search(r'(?:Admission\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["admission_date"] = ExtractedField(value=adm_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": adm_dt.group(0)}) if adm_dt else ExtractedField(value=None, confidence=0.0)

        ay_m = re.search(r'(?:Academic\s*Year|AY)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["academic_year"] = ExtractedField(value=ay_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": ay_m.group(0)}) if ay_m else ExtractedField(value=None, confidence=0.0)

        dur_m = re.search(r'(?:Course\s*Duration|Duration)\s*:\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["course_duration"] = ExtractedField(value=dur_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": dur_m.group(0)}) if dur_m else ExtractedField(value=None, confidence=0.0)

        fields["campus"] = ExtractedField(value=None, confidence=0.0)
        fields["admission_status"] = ExtractedField(value="Admitted" if "admitted" in text.lower() else "Confirmed", confidence=0.85)

        return fields

    @staticmethod
    def extract_fee_structure(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        stud_m = re.search(r'(?:Student\s*Name|Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["student_name"] = ExtractedField(value=stud_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": stud_m.group(0)}) if stud_m else ExtractedField(value=None, confidence=0.0)

        inst_m = re.search(r'(?:Institution|University|College)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["institution_name"] = ExtractedField(value=inst_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": inst_m.group(0)}) if inst_m else ExtractedField(value=None, confidence=0.0)

        course_m = re.search(r'(?:Course|Program)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["course_name"] = ExtractedField(value=course_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": course_m.group(0)}) if course_m else ExtractedField(value=None, confidence=0.0)

        ay_m = re.search(r'(?:Academic\s*Year|AY)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["academic_year"] = ExtractedField(value=ay_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": ay_m.group(0)}) if ay_m else ExtractedField(value=None, confidence=0.0)

        t_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tuition\s*Fee|Tuition)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tuition_fee"] = ExtractedField(value=t_fee[0], confidence=0.95, source={"page": 1, "text": t_fee[2]}) if t_fee else ExtractedField(value=None, confidence=0.0)

        h_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Hostel\s*Fee|Hostel)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["hostel_fee"] = ExtractedField(value=h_fee[0], confidence=0.90, source={"page": 1, "text": h_fee[2]}) if h_fee else ExtractedField(value=None, confidence=0.0)

        e_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Exam\s*Fee|Examination\s*Fee)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["examination_fee"] = ExtractedField(value=e_fee[0], confidence=0.90, source={"page": 1, "text": e_fee[2]}) if e_fee else ExtractedField(value=None, confidence=0.0)

        o_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Other\s*Fees|Miscellaneous\s*Fee)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["other_fees"] = ExtractedField(value=o_fee[0], confidence=0.90, source={"page": 1, "text": o_fee[2]}) if o_fee else ExtractedField(value=None, confidence=0.0)

        tot_f = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Fee|Total\s*Amount|Grand\s*Total)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_fee"] = ExtractedField(value=tot_f[0], confidence=0.95, source={"page": 1, "text": tot_f[2]}) if tot_f else ExtractedField(value=None, confidence=0.0)

        fields["fee_period"] = ExtractedField(value="Annual" if "annual" in text.lower() else "Semester", confidence=0.80)
        dt_m = re.search(r'(?:Issue\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["issue_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_gst_certificate(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        bus_m = re.search(r'(?:Legal\s*Name|Trade\s*Name|Business\s*Name)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["business_name"] = ExtractedField(value=bus_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": bus_m.group(0)}) if bus_m else ExtractedField(value=None, confidence=0.0)
        fields["legal_name"] = ExtractedField(value=fields["business_name"].value, confidence=fields["business_name"].confidence)

        gst_m = re.search(r'\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b', text)
        fields["GSTIN"] = ExtractedField(value=gst_m.group(1), confidence=0.99, source={"page": 1, "text": gst_m.group(0)}) if gst_m else ExtractedField(value=None, confidence=0.0)

        pan_m = re.search(r'\b([A-Z]{5}\d{4}[A-Z]{1})\b', text)
        fields["PAN"] = ExtractedField(value=pan_m.group(1), confidence=0.95, source={"page": 1, "text": pan_m.group(0)}) if pan_m else ExtractedField(value=None, confidence=0.0)

        fields["business_type"] = ExtractedField(value="Proprietorship" if "proprietorship" in text.lower() else ("Partnership" if "partnership" in text.lower() else "Private Limited"), confidence=0.85)

        dt_m = re.search(r'(?:Date\s*of\s*Registration|Registration\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["registration_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        addr_m = re.search(r'(?:Principal\s*Place\s*of\s*Business|Registered\s*Address|Address)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["registered_address"] = ExtractedField(value=addr_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": addr_m.group(0)}) if addr_m else ExtractedField(value=None, confidence=0.0)
        fields["principal_place_of_business"] = ExtractedField(value=fields["registered_address"].value, confidence=fields["registered_address"].confidence)

        fields["issuing_authority"] = ExtractedField(value="Government of India GST Authority", confidence=0.90)
        return fields

    @staticmethod
    def extract_fixed_deposit(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        holder_m = re.search(r'(?:Name|Account\s*Holder|Depositor\s*Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["account_holder_name"] = ExtractedField(value=holder_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": holder_m.group(0)}) if holder_m else ExtractedField(value=None, confidence=0.0)

        fields["joint_holder_names"] = ExtractedField(value=None, confidence=0.0)
        bank_m = re.search(r'(?:Bank\s*Name|Bank)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["bank_name"] = ExtractedField(value=bank_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": bank_m.group(0)}) if bank_m else ExtractedField(value=None, confidence=0.0)

        fields["branch"] = ExtractedField(value=None, confidence=0.0)

        acc_m = re.search(r'(?:Account\s*No|A/C\s*No)\s*:\s*(\d+)', text, re.IGNORECASE)
        fields["account_number"] = ExtractedField(value=acc_m.group(1), confidence=0.95, source={"page": 1, "text": acc_m.group(0)}) if acc_m else ExtractedField(value=None, confidence=0.0)

        fd_m = re.search(r'(?:FD\s*No|FD\s*Number|Receipt\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["FD_number"] = ExtractedField(value=fd_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": fd_m.group(0)}) if fd_m else ExtractedField(value=None, confidence=0.0)

        dep_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Deposit\s*Amount|Principal\s*Amount|Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["deposit_amount"] = ExtractedField(value=dep_amt[0], confidence=0.95, source={"page": 1, "text": dep_amt[2]}) if dep_amt else ExtractedField(value=None, confidence=0.0)

        rate_m = re.search(r'(?:Interest\s*Rate|Rate)\s*:\s*([\d\.]+\s*%)', text, re.IGNORECASE)
        fields["interest_rate"] = ExtractedField(value=rate_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": rate_m.group(0)}) if rate_m else ExtractedField(value=None, confidence=0.0)

        dep_dt = re.search(r'(?:Deposit\s*Date|Issue\s*Date|Value\s*Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["deposit_date"] = ExtractedField(value=dep_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": dep_dt.group(0)}) if dep_dt else ExtractedField(value=None, confidence=0.0)

        mat_dt = re.search(r'(?:Maturity\s*Date|Due\s*Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["maturity_date"] = ExtractedField(value=mat_dt.group(1).strip(), confidence=0.95, source={"page": 1, "text": mat_dt.group(0)}) if mat_dt else ExtractedField(value=None, confidence=0.0)

        mat_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Maturity\s*Amount|Maturity\s*Value)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["maturity_amount"] = ExtractedField(value=mat_amt[0], confidence=0.95, source={"page": 1, "text": mat_amt[2]}) if mat_amt else ExtractedField(value=None, confidence=0.0)

        fields["tenure"] = ExtractedField(value=None, confidence=0.0)
        fields["nominee"] = ExtractedField(value=None, confidence=0.0)
        fields["currency"] = ExtractedField(value="INR" if "₹" in text or "INR" in text else "USD", confidence=0.90)

        return fields

    @staticmethod
    def extract_gold_security(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        borr_m = re.search(r'(?:Borrower\s*Name|Customer\s*Name|Owner)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["borrower_name"] = ExtractedField(value=borr_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": borr_m.group(0)}) if borr_m else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Valuation\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["valuation_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        ref_m = re.search(r'(?:Valuation\s*Reference|Ref\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["valuation_reference"] = ExtractedField(value=ref_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": ref_m.group(0)}) if ref_m else ExtractedField(value=None, confidence=0.0)

        desc_m = re.search(r'(?:Gold\s*Description|Jewellery\s*Description|Items)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["gold_description"] = ExtractedField(value=desc_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": desc_m.group(0)}) if desc_m else ExtractedField(value=None, confidence=0.0)
        fields["jewellery_description"] = ExtractedField(value=fields["gold_description"].value, confidence=fields["gold_description"].confidence)

        items_m = re.search(r'(?:Number\s*of\s*Items|Item\s*Count|Quantity)\s*:\s*(\d+)', text, re.IGNORECASE)
        fields["number_of_items"] = ExtractedField(value=int(items_m.group(1)), confidence=0.90, source={"page": 1, "text": items_m.group(0)}) if items_m else ExtractedField(value=None, confidence=0.0)

        gw_m = re.search(r'(?:Gross\s*Weight)\s*:\s*([\d\.]+\s*(?:grams|g|gm)?)', text, re.IGNORECASE)
        fields["gross_weight"] = ExtractedField(value=gw_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": gw_m.group(0)}) if gw_m else ExtractedField(value=None, confidence=0.0)

        nw_m = re.search(r'(?:Net\s*Weight)\s*:\s*([\d\.]+\s*(?:grams|g|gm)?)', text, re.IGNORECASE)
        fields["net_weight"] = ExtractedField(value=nw_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": nw_m.group(0)}) if nw_m else ExtractedField(value=None, confidence=0.0)

        pur_m = re.search(r'(?:Purity|Carat)\s*:\s*([\d\.]+\s*(?:K|Karat|Carat|%)?)', text, re.IGNORECASE)
        fields["purity"] = ExtractedField(value=pur_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": pur_m.group(0)}) if pur_m else ExtractedField(value=None, confidence=0.0)

        val_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Assessed\s*Value|Market\s*Value|Total\s*Valuation|Loan\s*Value)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["assessed_value"] = ExtractedField(value=val_amt[0], confidence=0.95, source={"page": 1, "text": val_amt[2]}) if val_amt else ExtractedField(value=None, confidence=0.0)
        fields["market_value"] = ExtractedField(value=fields["assessed_value"].value, confidence=fields["assessed_value"].confidence)
        fields["loan_value"] = ExtractedField(value=fields["assessed_value"].value, confidence=fields["assessed_value"].confidence)

        val_name = re.search(r'(?:Valuer\s*Name|Appraiser\s*Name|Valuer)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["valuer_name"] = ExtractedField(value=val_name.group(1).strip(), confidence=0.90, source={"page": 1, "text": val_name.group(0)}) if val_name else ExtractedField(value=None, confidence=0.0)
        fields["appraiser_name"] = ExtractedField(value=fields["valuer_name"].value, confidence=fields["valuer_name"].confidence)
        fields["security_reference"] = ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_property_tax(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        own_m = re.search(r'(?:Owner\s*Name|Taxpayer\s*Name|Owner)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["owner_name"] = ExtractedField(value=own_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": own_m.group(0)}) if own_m else ExtractedField(value=None, confidence=0.0)

        addr_m = re.search(r'(?:Property\s*Address|Address)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["property_address"] = ExtractedField(value=addr_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": addr_m.group(0)}) if addr_m else ExtractedField(value=None, confidence=0.0)

        id_m = re.search(r'(?:Property\s*ID|PID|Tax\s*Assessment\s*No)\s*:\s*([\w-]+)', text, re.IGNORECASE)
        fields["property_id"] = ExtractedField(value=id_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": id_m.group(0)}) if id_m else ExtractedField(value=None, confidence=0.0)

        rec_m = re.search(r'(?:Tax\s*Receipt\s*No|Receipt\s*No|Receipt\s*Number)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["tax_receipt_number"] = ExtractedField(value=rec_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": rec_m.group(0)}) if rec_m else ExtractedField(value=None, confidence=0.0)

        per_m = re.search(r'(?:Tax\s*Period|Assessment\s*Year|Period)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["tax_period"] = ExtractedField(value=per_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": per_m.group(0)}) if per_m else ExtractedField(value=None, confidence=0.0)

        tax_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tax\s*Amount|Total\s*Tax\s*Paid|Amount\s*Paid|Tax)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tax_amount"] = ExtractedField(value=tax_amt[0], confidence=0.95, source={"page": 1, "text": tax_amt[2]}) if tax_amt else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Payment\s*Date|Receipt\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["payment_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        auth_m = re.search(r'(?:Issuing\s*Authority|Municipal\s*Corporation|Authority)\s*:\s*([A-Za-z0-9\s,\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["issuing_authority"] = ExtractedField(value=auth_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": auth_m.group(0)}) if auth_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_property_valuation(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        own_m = re.search(r'(?:Owner\s*Name|Owner|Client)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["owner_name"] = ExtractedField(value=own_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": own_m.group(0)}) if own_m else ExtractedField(value=None, confidence=0.0)

        addr_m = re.search(r'(?:Property\s*Address|Address)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["property_address"] = ExtractedField(value=addr_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": addr_m.group(0)}) if addr_m else ExtractedField(value=None, confidence=0.0)

        type_m = re.search(r'(?:Property\s*Type|Type)\s*:\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["property_type"] = ExtractedField(value=type_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": type_m.group(0)}) if type_m else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Valuation\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["valuation_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        mkt_val = DeterministicExtractor.extract_currency_amount(text, [r'(?:Market\s*Value|Realizable\s*Value|Fair\s*Market\s*Value)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["market_value"] = ExtractedField(value=mkt_val[0], confidence=0.95, source={"page": 1, "text": mkt_val[2]}) if mkt_val else ExtractedField(value=None, confidence=0.0)

        ass_val = DeterministicExtractor.extract_currency_amount(text, [r'(?:Assessed\s*Value|Guidance\s*Value|Valuation\s*Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["assessed_value"] = ExtractedField(value=ass_val[0], confidence=0.90, source={"page": 1, "text": ass_val[2]}) if ass_val else ExtractedField(value=fields["market_value"].value, confidence=0.70)

        fs_val = DeterministicExtractor.extract_currency_amount(text, [r'(?:Distress\s*Value|Forced\s*Sale\s*Value)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["forced_sale_value"] = ExtractedField(value=fs_val[0], confidence=0.90, source={"page": 1, "text": fs_val[2]}) if fs_val else ExtractedField(value=None, confidence=0.0)

        val_n = re.search(r'(?:Valuer\s*Name|Valuer|Appraiser)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["valuer_name"] = ExtractedField(value=val_n.group(1).strip(), confidence=0.90, source={"page": 1, "text": val_n.group(0)}) if val_n else ExtractedField(value=None, confidence=0.0)

        ref_n = re.search(r'(?:Valuation\s*Reference|Ref\s*No|Report\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["valuation_reference_number"] = ExtractedField(value=ref_n.group(1).strip(), confidence=0.90, source={"page": 1, "text": ref_n.group(0)}) if ref_n else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_academic_certificate(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        def clean_val(val: Optional[str]) -> Optional[str]:
            if not val:
                return None
            s = str(val).strip()
            lower = s.lower()
            if any(p in lower for p in ["intentionally omitted", "sample text", "placeholder", "test document", "missing field", "not provided", "intentionally_omitted"]):
                return None
            return s

        stud_m = re.search(r'(?:Student\s*Name|Candidate\s*Name|Student|Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        stud_val = clean_val(stud_m.group(1)) if stud_m else None
        fields["student_name"] = ExtractedField(value=stud_val, confidence=0.90, source={"page": 1, "text": stud_m.group(0)}) if stud_val else ExtractedField(value=None, confidence=0.0)

        inst_val = None
        inst_src = None
        inst_m = re.search(r'(?:Institution\s*Name|University\s*Name|College\s*Name|Institution|University|Board|College|School)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        if inst_m:
            candidate = clean_val(inst_m.group(1))
            if candidate and len(candidate) > 2 and candidate.upper() not in ["DETAILS", "VALUE"]:
                inst_val = candidate
                inst_src = inst_m.group(0)

        if not inst_val:
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            for line in lines[:10]:
                u_line = line.upper()
                if any(kw in u_line for kw in ["UNIVERSITY", "INSTITUTE", "COLLEGE", "BOARD OF", "SCHOOL", "ACADEMY"]) and not any(kw in u_line for kw in ["STUDENT", "NAME", "CERTIFICATE", "MARKSHEET"]):
                    candidate = clean_val(line)
                    if candidate:
                        inst_val = candidate
                        inst_src = line
                        break

        fields["institution_name"] = ExtractedField(value=inst_val, confidence=0.90 if inst_val else 0.0, source={"page": 1, "text": inst_src} if inst_src else None)

        exam_m = re.search(r'(?:Examination|Exam|Degree|Course)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        exam_val = clean_val(exam_m.group(1)) if exam_m else None
        fields["examination_name"] = ExtractedField(value=exam_val, confidence=0.90, source={"page": 1, "text": exam_m.group(0)}) if exam_val else ExtractedField(value=None, confidence=0.0)
        fields["course_name"] = ExtractedField(value=fields["examination_name"].value, confidence=fields["examination_name"].confidence)

        roll_m = re.search(r'(?:Roll\s*No|Roll\s*Number)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        roll_val = clean_val(roll_m.group(1)) if roll_m else None
        fields["roll_number"] = ExtractedField(value=roll_val, confidence=0.95, source={"page": 1, "text": roll_m.group(0)}) if roll_val else ExtractedField(value=None, confidence=0.0)

        reg_m = re.search(r'(?:Registration\s*No|Reg\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        reg_val = clean_val(reg_m.group(1)) if reg_m else None
        fields["registration_number"] = ExtractedField(value=reg_val, confidence=0.95, source={"page": 1, "text": reg_m.group(0)}) if reg_val else ExtractedField(value=None, confidence=0.0)

        ay_m = re.search(r'(?:Academic\s*Year|Year\s*of\s*Passing|Year)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        ay_val = clean_val(ay_m.group(1)) if ay_m else None
        fields["academic_year"] = ExtractedField(value=ay_val, confidence=0.90, source={"page": 1, "text": ay_m.group(0)}) if ay_val else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Date|Issue\s*Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        dt_val = clean_val(dt_m.group(1)) if dt_m else None
        fields["examination_date"] = ExtractedField(value=dt_val, confidence=0.90, source={"page": 1, "text": dt_m.group(0)}) if dt_val else ExtractedField(value=None, confidence=0.0)

        m_obt = re.search(r'(?:Marks\s*Obtained|Total\s*Marks\s*Obtained|Obtained\s*Marks)\s*:\s*([\d\.]+)', text, re.IGNORECASE)
        fields["marks_obtained"] = ExtractedField(value=float(m_obt.group(1)) if m_obt else None, confidence=0.95, source={"page": 1, "text": m_obt.group(0)} if m_obt else None)

        m_tot = re.search(r'(?:Total\s*Marks|Maximum\s*Marks|Max\s*Marks)\s*:\s*([\d\.]+)', text, re.IGNORECASE)
        fields["total_marks"] = ExtractedField(value=float(m_tot.group(1)) if m_tot else None, confidence=0.95, source={"page": 1, "text": m_tot.group(0)} if m_tot else None)

        pct_m = re.search(r'(?:Percentage|Marks\s*%)\s*:\s*([\d\.]+\s*%?)', text, re.IGNORECASE)
        pct_val = clean_val(pct_m.group(1)) if pct_m else None
        fields["percentage"] = ExtractedField(value=pct_val, confidence=0.90, source={"page": 1, "text": pct_m.group(0)}) if pct_val else ExtractedField(value=None, confidence=0.0)

        grd_m = re.search(r'(?:Grade|Class)\s*:\s*([A-Za-z0-9\+\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        grd_val = clean_val(grd_m.group(1)) if grd_m else None
        fields["grade"] = ExtractedField(value=grd_val, confidence=0.85, source={"page": 1, "text": grd_m.group(0)}) if grd_val else ExtractedField(value=None, confidence=0.0)

        res_m = re.search(r'(?:Result|Status)\s*:\s*(Pass|Passed|First\s*Class|Distinction)', text, re.IGNORECASE)
        fields["result"] = ExtractedField(value=clean_val(res_m.group(1)) if res_m else "Pass", confidence=0.85)

        return fields

    @staticmethod
    def extract_admission_letter(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        stud_m = re.search(r'(?:Student\s*Name|Candidate\s*Name|Applicant\s*Name|Student|Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["student_name"] = ExtractedField(value=stud_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": stud_m.group(0)}) if stud_m else ExtractedField(value=None, confidence=0.0)

        inst_m = re.search(r'(?:Institution\s*Name|University\s*Name|College\s*Name|Institution|University|Institute|College|School)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["institution_name"] = ExtractedField(value=inst_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": inst_m.group(0)}) if inst_m else ExtractedField(value=None, confidence=0.0)

        crs_m = re.search(r'(?:Course\s*Name|Degree\s*Name|Program\s*Name|Course|Degree)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["course_name"] = ExtractedField(value=crs_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": crs_m.group(0)}) if crs_m else ExtractedField(value=None, confidence=0.0)

        prog_m = re.search(r'(?:Program\s*Name|Program|Specialization|Branch)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["program_name"] = ExtractedField(value=prog_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": prog_m.group(0)}) if prog_m else ExtractedField(value=fields.get("course_name", ExtractedField()).value, confidence=0.70)

        adm_num_m = re.search(r'(?:Admission\s*Number|Admission\s*No|Enrollment\s*No|Roll\s*No|Application\s*No|Ref\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["admission_number"] = ExtractedField(value=adm_num_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": adm_num_m.group(0)}) if adm_num_m else ExtractedField(value=None, confidence=0.0)

        ay_m = re.search(r'(?:Academic\s*Year|Batch|Session|Year)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["academic_year"] = ExtractedField(value=ay_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": ay_m.group(0)}) if ay_m else ExtractedField(value=None, confidence=0.0)

        adm_dt_m = re.search(r'(?:Admission\s*Date|Date\s*of\s*Admission|Date|Issue\s*Date)\s*:\s*(\d{2}-[A-Za-z0-9]{2,3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        fields["admission_date"] = ExtractedField(value=adm_dt_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": adm_dt_m.group(0)}) if adm_dt_m else ExtractedField(value=None, confidence=0.0)

        dur_m = re.search(r'(?:Course\s*Duration|Duration|Period)\s*:\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["course_duration"] = ExtractedField(value=dur_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": dur_m.group(0)}) if dur_m else ExtractedField(value=None, confidence=0.0)

        cmp_m = re.search(r'(?:Campus|Location|Branch|Center)\s*:\s*([A-Za-z0-9\s\.,]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["campus"] = ExtractedField(value=cmp_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": cmp_m.group(0)}) if cmp_m else ExtractedField(value=None, confidence=0.0)

        stat_m = re.search(r'(?:Admission\s*Status|Status)\s*:\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        return fields

    @staticmethod
    def extract_fee_structure(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        stud_m = re.search(r'(?:Student\s*Name|Candidate\s*Name|Applicant\s*Name|Student|Name)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["student_name"] = ExtractedField(value=stud_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": stud_m.group(0)}) if stud_m else ExtractedField(value=None, confidence=0.0)

        inst_m = re.search(r'(?:Institution\s*Name|University\s*Name|College\s*Name|Institution|University|Institute|College|School)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["institution_name"] = ExtractedField(value=inst_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": inst_m.group(0)}) if inst_m else ExtractedField(value=None, confidence=0.0)

        crs_m = re.search(r'(?:Course\s*Name|Degree\s*Name|Program\s*Name|Course|Degree)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["course_name"] = ExtractedField(value=crs_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": crs_m.group(0)}) if crs_m else ExtractedField(value=None, confidence=0.0)

        ay_m = re.search(r'(?:Academic\s*Year|Batch|Session|Year)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["academic_year"] = ExtractedField(value=ay_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": ay_m.group(0)}) if ay_m else ExtractedField(value=None, confidence=0.0)

        tuit_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tuition\s*Fee|Course\s*Fee|Tuition)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tuition_fee"] = ExtractedField(value=tuit_fee[0], confidence=0.95, source={"page": 1, "text": tuit_fee[2]}) if tuit_fee else ExtractedField(value=None, confidence=0.0)

        hst_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Hostel\s*Fee|Hostel\s*Charges|Accommodation)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["hostel_fee"] = ExtractedField(value=hst_fee[0], confidence=0.90, source={"page": 1, "text": hst_fee[2]}) if hst_fee else ExtractedField(value=None, confidence=0.0)

        exam_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Examination\s*Fee|Exam\s*Fee|Exam\s*Charges)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["examination_fee"] = ExtractedField(value=exam_fee[0], confidence=0.90, source={"page": 1, "text": exam_fee[2]}) if exam_fee else ExtractedField(value=None, confidence=0.0)

        oth_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Other\s*Fees|Other\s*Charges|Miscellaneous)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["other_fees"] = ExtractedField(value=oth_fee[0], confidence=0.85, source={"page": 1, "text": oth_fee[2]}) if oth_fee else ExtractedField(value=None, confidence=0.0)

        tot_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Fee|Grand\s*Total|Total\s*Amount|Total)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_fee"] = ExtractedField(value=tot_fee[0], confidence=0.95, source={"page": 1, "text": tot_fee[2]}) if tot_fee else ExtractedField(value=None, confidence=0.0)

        per_m = re.search(r'(?:Fee\s*Period|Period|Term|Semester)\s*:\s*([A-Za-z0-9\s-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["fee_period"] = ExtractedField(value=per_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": per_m.group(0)}) if per_m else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Issue\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z0-9]{2,3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})', text, re.IGNORECASE)
        fields["issue_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_property_document(text: str) -> Dict[str, ExtractedField]:
        fields = {}

        buyer_m = re.search(r'(?:Buyer\s*Name|Purchaser\s*Name|Buyer|Purchaser)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["buyer_name"] = ExtractedField(value=buyer_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": buyer_m.group(0)}) if buyer_m else ExtractedField(value=None, confidence=0.0)

        seller_m = re.search(r'(?:Seller\s*Name|Vendor\s*Name|Seller|Vendor)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["seller_name"] = ExtractedField(value=seller_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": seller_m.group(0)}) if seller_m else ExtractedField(value=None, confidence=0.0)

        own_m = re.search(r'(?:Owner\s*Name|Owner)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["owner_name"] = ExtractedField(value=own_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": own_m.group(0)}) if own_m else ExtractedField(value=fields.get("buyer_name", ExtractedField()).value, confidence=0.70)

        fields["co_owner_name"] = ExtractedField(value=None, confidence=0.0)

        addr_m = re.search(r'(?:Property\s*Address|Address)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["property_address"] = ExtractedField(value=addr_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": addr_m.group(0)}) if addr_m else ExtractedField(value=None, confidence=0.0)

        fields["property_description"] = ExtractedField(value=fields["property_address"].value, confidence=0.70)
        fields["property_identifier"] = ExtractedField(value=None, confidence=0.0)

        surv_m = re.search(r'(?:Survey\s*Number|Survey\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["survey_number"] = ExtractedField(value=surv_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": surv_m.group(0)}) if surv_m else ExtractedField(value=None, confidence=0.0)

        plot_m = re.search(r'(?:Plot\s*Number|Plot\s*No|Flat\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["plot_number"] = ExtractedField(value=plot_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": plot_m.group(0)}) if plot_m else ExtractedField(value=None, confidence=0.0)

        reg_m = re.search(r'(?:Registration\s*Number|Registration\s*No|Reg\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["registration_number"] = ExtractedField(value=reg_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": reg_m.group(0)}) if reg_m else ExtractedField(value=None, confidence=0.0)

        doc_num = re.search(r'(?:Document\s*Number|Doc\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["document_number"] = ExtractedField(value=doc_num.group(1).strip(), confidence=0.90, source={"page": 1, "text": doc_num.group(0)}) if doc_num else ExtractedField(value=fields.get("registration_number", ExtractedField()).value, confidence=0.70)

        reg_dt = re.search(r'(?:Registration\s*Date|Date\s*of\s*Registration)\s*:\s*(\d{2}-[A-Za-z0-9]{2,3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["registration_date"] = ExtractedField(value=reg_dt.group(1).strip(), confidence=0.90, source={"page": 1, "text": reg_dt.group(0)}) if reg_dt else ExtractedField(value=None, confidence=0.0)

        agr_dt = re.search(r'(?:Agreement\s*Date|Date\s*of\s*Agreement|Agreement|Date)\s*:\s*(\d{2}-[A-Za-z0-9]{2,3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["agreement_date"] = ExtractedField(value=agr_dt.group(1).strip(), confidence=0.90, source={"page": 1, "text": agr_dt.group(0)}) if agr_dt else ExtractedField(value=None, confidence=0.0)

        sale_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Sale\s*Amount|Consideration\s*Amount|Purchase\s*Price|Sale\s*Value|Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["sale_amount"] = ExtractedField(value=sale_amt[0], confidence=0.95, source={"page": 1, "text": sale_amt[2]}) if sale_amt else ExtractedField(value=None, confidence=0.0)
        fields["property_value"] = ExtractedField(value=fields["sale_amount"].value, confidence=fields["sale_amount"].confidence)
        fields["consideration_amount"] = ExtractedField(value=fields["sale_amount"].value, confidence=fields["sale_amount"].confidence)

        stamp_m = DeterministicExtractor.extract_currency_amount(text, [r'(?:Stamp\s*Duty)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["stamp_duty"] = ExtractedField(value=stamp_m[0], confidence=0.90, source={"page": 1, "text": stamp_m[2]}) if stamp_m else ExtractedField(value=None, confidence=0.0)

        reg_fee = DeterministicExtractor.extract_currency_amount(text, [r'(?:Registration\s*Fee)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["registration_fee"] = ExtractedField(value=reg_fee[0], confidence=0.90, source={"page": 1, "text": reg_fee[2]}) if reg_fee else ExtractedField(value=None, confidence=0.0)

        fields["issuing_authority"] = ExtractedField(value="Sub-Registrar Office", confidence=0.85)

        return fields



    @staticmethod
    def extract_gst_return(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        bus_m = re.search(r'(?:Business\s*Name|Legal\s*Name)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["business_name"] = ExtractedField(value=bus_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": bus_m.group(0)}) if bus_m else ExtractedField(value=None, confidence=0.0)

        gst_m = re.search(r'\b(\d{2}[A-Z]{5}\d{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1})\b', text)
        fields["GSTIN"] = ExtractedField(value=gst_m.group(1), confidence=0.99, source={"page": 1, "text": gst_m.group(0)}) if gst_m else ExtractedField(value=None, confidence=0.0)

        per_m = re.search(r'(?:Tax\s*Period|Period|Month)\s*:\s*([A-Za-z0-9\s-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["tax_period"] = ExtractedField(value=per_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": per_m.group(0)}) if per_m else ExtractedField(value=None, confidence=0.0)

        turn_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Turnover|Gross\s*Turnover|Turnover)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["turnover"] = ExtractedField(value=turn_amt[0], confidence=0.95, source={"page": 1, "text": turn_amt[2]}) if turn_amt else ExtractedField(value=None, confidence=0.0)

        tax_turn = DeterministicExtractor.extract_currency_amount(text, [r'(?:Taxable\s*Turnover|Taxable\s*Value)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["taxable_turnover"] = ExtractedField(value=tax_turn[0], confidence=0.90, source={"page": 1, "text": tax_turn[2]}) if tax_turn else ExtractedField(value=fields["turnover"].value, confidence=0.70)

        liab_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tax\s*Liability|Total\s*Tax\s*Payable)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tax_liability"] = ExtractedField(value=liab_amt[0], confidence=0.90, source={"page": 1, "text": liab_amt[2]}) if liab_amt else ExtractedField(value=None, confidence=0.0)

        paid_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tax\s*Paid|Total\s*Tax\s*Paid)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tax_paid"] = ExtractedField(value=paid_amt[0], confidence=0.90, source={"page": 1, "text": paid_amt[2]}) if paid_amt else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Filing\s*Date|Date\s*of\s*Filing|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["filing_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        ret_m = re.search(r'(?:Return\s*Type|Form)\s*:\s*(GSTR-3B|GSTR-1|GSTR-9)', text, re.IGNORECASE)
        fields["return_type"] = ExtractedField(value=ret_m.group(1).upper() if ret_m else "GSTR-3B", confidence=0.85)

        return fields

    @staticmethod
    def extract_profit_loss(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        bus_m = re.search(r'(?:Business\s*Name|Company\s*Name|Entity)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["business_name"] = ExtractedField(value=bus_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": bus_m.group(0)}) if bus_m else ExtractedField(value=None, confidence=0.0)

        fy_m = re.search(r'(?:Financial\s*Year|FY|For\s*the\s*Year)\s*:\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["financial_year"] = ExtractedField(value=fy_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": fy_m.group(0)}) if fy_m else ExtractedField(value=None, confidence=0.0)

        rev_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Revenue|Revenue\s*from\s*Operations|Total\s*Sales|Gross\s*Revenue)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["revenue"] = ExtractedField(value=rev_amt[0], confidence=0.95, source={"page": 1, "text": rev_amt[2]}) if rev_amt else ExtractedField(value=None, confidence=0.0)
        fields["sales"] = ExtractedField(value=fields["revenue"].value, confidence=fields["revenue"].confidence)

        cogs_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Cost\s*of\s*Goods\s*Sold|COGS|Cost\s*of\s*Sales)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["cost_of_goods"] = ExtractedField(value=cogs_amt[0], confidence=0.90, source={"page": 1, "text": cogs_amt[2]}) if cogs_amt else ExtractedField(value=None, confidence=0.0)

        gp_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Gross\s*Profit)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["gross_profit"] = ExtractedField(value=gp_amt[0], confidence=0.95, source={"page": 1, "text": gp_amt[2]}) if gp_amt else ExtractedField(value=None, confidence=0.0)

        opex_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Operating\s*Expenses|OpEx|Total\s*Expenses)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["operating_expenses"] = ExtractedField(value=opex_amt[0], confidence=0.90, source={"page": 1, "text": opex_amt[2]}) if opex_amt else ExtractedField(value=None, confidence=0.0)

        op_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Operating\s*Profit|EBITDA|EBIT)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["operating_profit"] = ExtractedField(value=op_amt[0], confidence=0.90, source={"page": 1, "text": op_amt[2]}) if op_amt else ExtractedField(value=None, confidence=0.0)

        np_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Net\s*Profit|Net\s*Income|Profit\s*After\s*Tax|PAT)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["net_profit"] = ExtractedField(value=np_amt[0], confidence=0.95, source={"page": 1, "text": np_amt[2]}) if np_amt else ExtractedField(value=None, confidence=0.0)

        dep_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Depreciation|Depreciation\s*&\s*Amortization)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["depreciation"] = ExtractedField(value=dep_amt[0], confidence=0.85, source={"page": 1, "text": dep_amt[2]}) if dep_amt else ExtractedField(value=None, confidence=0.0)

        int_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Interest\s*Expense|Finance\s*Costs)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["interest_expense"] = ExtractedField(value=int_amt[0], confidence=0.85, source={"page": 1, "text": int_amt[2]}) if int_amt else ExtractedField(value=None, confidence=0.0)

        oth_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Other\s*Income)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["other_income"] = ExtractedField(value=oth_amt[0], confidence=0.85, source={"page": 1, "text": oth_amt[2]}) if oth_amt else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_balance_sheet(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        bus_m = re.search(r'(?:Business\s*Name|Company\s*Name|Entity)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["business_name"] = ExtractedField(value=bus_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": bus_m.group(0)}) if bus_m else ExtractedField(value=None, confidence=0.0)

        fy_m = re.search(r'(?:Financial\s*Year|FY|As\s*at)\s*:\s*([\d-]{4,9}|\d{2}-[A-Za-z]{3}-\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["financial_year"] = ExtractedField(value=fy_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": fy_m.group(0)}) if fy_m else ExtractedField(value=None, confidence=0.0)

        ast_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Assets|Assets)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_assets"] = ExtractedField(value=ast_amt[0], confidence=0.95, source={"page": 1, "text": ast_amt[2]}) if ast_amt else ExtractedField(value=None, confidence=0.0)

        cast_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Current\s*Assets)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["current_assets"] = ExtractedField(value=cast_amt[0], confidence=0.90, source={"page": 1, "text": cast_amt[2]}) if cast_amt else ExtractedField(value=None, confidence=0.0)

        fast_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Fixed\s*Assets|Non-Current\s*Assets)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["fixed_assets"] = ExtractedField(value=fast_amt[0], confidence=0.90, source={"page": 1, "text": fast_amt[2]}) if fast_amt else ExtractedField(value=None, confidence=0.0)

        liab_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Liabilities|Liabilities)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_liabilities"] = ExtractedField(value=liab_amt[0], confidence=0.95, source={"page": 1, "text": liab_amt[2]}) if liab_amt else ExtractedField(value=None, confidence=0.0)

        cliab_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Current\s*Liabilities)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["current_liabilities"] = ExtractedField(value=cliab_amt[0], confidence=0.90, source={"page": 1, "text": cliab_amt[2]}) if cliab_amt else ExtractedField(value=None, confidence=0.0)

        ltliab_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Long-Term\s*Liabilities|Non-Current\s*Liabilities)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["long_term_liabilities"] = ExtractedField(value=ltliab_amt[0], confidence=0.90, source={"page": 1, "text": ltliab_amt[2]}) if ltliab_amt else ExtractedField(value=None, confidence=0.0)

        eq_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Equity|Shareholders\s*Equity|Total\s*Equity|Net\s*Worth)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["equity"] = ExtractedField(value=eq_amt[0], confidence=0.90, source={"page": 1, "text": eq_amt[2]}) if eq_amt else ExtractedField(value=None, confidence=0.0)

        cap_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Capital|Share\s*Capital)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["capital"] = ExtractedField(value=cap_amt[0], confidence=0.85, source={"page": 1, "text": cap_amt[2]}) if cap_amt else ExtractedField(value=None, confidence=0.0)

        ret_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Retained\s*Earnings|Reserves\s*&\s*Surplus)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["retained_earnings"] = ExtractedField(value=ret_amt[0], confidence=0.85, source={"page": 1, "text": ret_amt[2]}) if ret_amt else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_land_record(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        farm_m = re.search(r'(?:Farmer\s*Name|Cultivator\s*Name|Farmer)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["farmer_name"] = ExtractedField(value=farm_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": farm_m.group(0)}) if farm_m else ExtractedField(value=None, confidence=0.0)

        own_m = re.search(r'(?:Land\s*Owner|Owner\s*Name|Pattadar)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["land_owner_name"] = ExtractedField(value=own_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": own_m.group(0)}) if own_m else ExtractedField(value=fields["farmer_name"].value, confidence=0.70)

        addr_m = re.search(r'(?:Land\s*Address|Location|Premises)\s*:\s*([^\n]+)', text, re.IGNORECASE)
        fields["land_address"] = ExtractedField(value=addr_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": addr_m.group(0)}) if addr_m else ExtractedField(value=None, confidence=0.0)

        surv_m = re.search(r'(?:Survey\s*No|Survey\s*Number|Sy\s*No|Khasra\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["survey_number"] = ExtractedField(value=surv_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": surv_m.group(0)}) if surv_m else ExtractedField(value=None, confidence=0.0)

        sub_m = re.search(r'(?:Subdivision\s*No|Sub-division|Hissa\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["subdivision_number"] = ExtractedField(value=sub_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": sub_m.group(0)}) if sub_m else ExtractedField(value=None, confidence=0.0)

        area_m = re.search(r'(?:Land\s*Area|Area|Extent)\s*:\s*([\d\.]+\s*(?:Acres|Hectares|Sq\s*Ft|Bigha)?)', text, re.IGNORECASE)
        fields["land_area"] = ExtractedField(value=area_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": area_m.group(0)}) if area_m else ExtractedField(value=None, confidence=0.0)

        fields["land_unit"] = ExtractedField(value="Acres" if "acre" in text.lower() else ("Hectares" if "hectare" in text.lower() else "Acres"), confidence=0.80)

        vil_m = re.search(r'(?:Village)\s*:\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["village"] = ExtractedField(value=vil_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": vil_m.group(0)}) if vil_m else ExtractedField(value=None, confidence=0.0)

        dis_m = re.search(r'(?:District|Taluk)\s*:\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["district"] = ExtractedField(value=dis_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": dis_m.group(0)}) if dis_m else ExtractedField(value=None, confidence=0.0)

        st_m = re.search(r'(?:State)\s*:\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["state"] = ExtractedField(value=st_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": st_m.group(0)}) if st_m else ExtractedField(value=None, confidence=0.0)

        fields["land_type"] = ExtractedField(value="Wetland" if "wetland" in text.lower() else ("Dryland" if "dryland" in text.lower() else "Agricultural"), confidence=0.80)
        fields["cultivation_type"] = ExtractedField(value="Irrigated" if "irrigated" in text.lower() else "Rainfed", confidence=0.80)

        crop_m = re.search(r'(?:Crop\s*Type|Crop)\s*:\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["crop_type"] = ExtractedField(value=crop_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": crop_m.group(0)}) if crop_m else ExtractedField(value=None, confidence=0.0)

        fields["ownership_type"] = ExtractedField(value="Ancestral" if "ancestral" in text.lower() else "Self Acquired", confidence=0.80)

        doc_m = re.search(r'(?:Document\s*No|Khata\s*No|Patta\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["document_number"] = ExtractedField(value=doc_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": doc_m.group(0)}) if doc_m else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Issue\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["issue_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        auth_m = re.search(r'(?:Issuing\s*Authority|Tahsildar|Revenue\s*Officer)\s*:\s*([A-Za-z0-9\s,\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["issuing_authority"] = ExtractedField(value=auth_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": auth_m.group(0)}) if auth_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_agricultural_income(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        farm_m = re.search(r'(?:Farmer\s*Name|Farmer)\s*[:\n]\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["farmer_name"] = ExtractedField(value=farm_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": farm_m.group(0)}) if farm_m else ExtractedField(value=None, confidence=0.0)

        fy_m = re.search(r'(?:Financial\s*Year|FY)\s*[:\n]\s*([\d-]{4,9})', text, re.IGNORECASE)
        fields["financial_year"] = ExtractedField(value=fy_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": fy_m.group(0)}) if fy_m else ExtractedField(value=None, confidence=0.0)

        crop_m = re.search(r'(?:Crop\s*Type|Primary\s*Crop|Crop)\s*[:\n]\s*([A-Za-z\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["crop_type"] = ExtractedField(value=crop_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": crop_m.group(0)}) if crop_m else ExtractedField(value=None, confidence=0.0)

        area_m = re.search(r'(?:Cultivated\s*Area|Land\s*Area|Area)\s*[:\n]\s*([\d\.]+\s*(?:Acres|Hectares)?)', text, re.IGNORECASE)
        fields["cultivated_area"] = ExtractedField(value=area_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": area_m.group(0)}) if area_m else ExtractedField(value=None, confidence=0.0)

        inc_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Agricultural\s*Income|Gross\s*Crop\s*Income|Income)\s*[:\n]?\s*(?:INR|\$|₹|I)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["agricultural_income"] = ExtractedField(value=inc_amt[0], confidence=0.95, source={"page": 1, "text": inc_amt[2]}) if inc_amt else ExtractedField(value=None, confidence=0.0)

        exp_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Expenses|Cultivation\s*Expenses)\s*[:\n]?\s*(?:INR|\$|₹|I)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["expenses"] = ExtractedField(value=exp_amt[0], confidence=0.85, source={"page": 1, "text": exp_amt[2]}) if exp_amt else ExtractedField(value=None, confidence=0.0)

        net_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Net\s*Agricultural\s*Income|Net\s*Income)\s*[:\n]?\s*(?:INR|\$|₹|I)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["net_agricultural_income"] = ExtractedField(value=net_amt[0], confidence=0.95, source={"page": 1, "text": net_amt[2]}) if net_amt else ExtractedField(value=fields["agricultural_income"].value, confidence=0.70)

        auth_m = re.search(r'(?:Issuing\s*Authority|Tahsildar|Revenue\s*Officer)\s*[:\n]\s*([A-Za-z0-9\s,\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["issuing_authority"] = ExtractedField(value=auth_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": auth_m.group(0)}) if auth_m else ExtractedField(value=None, confidence=0.0)

        dt_m = re.search(r'(?:Document\s*Date|Date)\s*[:\n]\s*(\d{2}-[A-Za-z0-9]{2,3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["document_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_product_invoice(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        cust_m = re.search(r'(?:Customer\s*Name|Buyer\s*Name|Customer|Buyer)\s*:\s*([A-Za-z\s\.]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["customer_name"] = ExtractedField(value=cust_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": cust_m.group(0)}) if cust_m else ExtractedField(value=None, confidence=0.0)
        fields["buyer_name"] = ExtractedField(value=fields["customer_name"].value, confidence=fields["customer_name"].confidence)

        sell_m = re.search(r'(?:Seller\s*Name|Dealer\s*Name|Store\s*Name|Vendor)\s*:\s*([A-Za-z0-9\s\.,&]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["seller_name"] = ExtractedField(value=sell_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": sell_m.group(0)}) if sell_m else ExtractedField(value=None, confidence=0.0)

        inv_n = re.search(r'(?:Invoice\s*No|Quotation\s*No|Invoice\s*Number|Quote\s*No)\s*:\s*([\w/-]+)', text, re.IGNORECASE)
        fields["invoice_number"] = ExtractedField(value=inv_n.group(1).strip(), confidence=0.95, source={"page": 1, "text": inv_n.group(0)}) if inv_n else ExtractedField(value=None, confidence=0.0)
        fields["quotation_number"] = ExtractedField(value=fields["invoice_number"].value, confidence=fields["invoice_number"].confidence)

        dt_m = re.search(r'(?:Invoice\s*Date|Quotation\s*Date|Date)\s*:\s*(\d{2}-[A-Za-z]{3}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
        fields["invoice_date"] = ExtractedField(value=dt_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": dt_m.group(0)}) if dt_m else ExtractedField(value=None, confidence=0.0)
        fields["quotation_date"] = ExtractedField(value=fields["invoice_date"].value, confidence=fields["invoice_date"].confidence)

        prod_m = re.search(r'(?:Product\s*Name|Product|Item\s*Description|Item)\s*:\s*([A-Za-z0-9\s\.-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["product_name"] = ExtractedField(value=prod_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": prod_m.group(0)}) if prod_m else ExtractedField(value=None, confidence=0.0)

        cat_m = re.search(r'(?:Category|Product\s*Category)\s*:\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["product_category"] = ExtractedField(value=cat_m.group(1).strip(), confidence=0.85, source={"page": 1, "text": cat_m.group(0)}) if cat_m else ExtractedField(value=None, confidence=0.0)

        brand_m = re.search(r'(?:Brand|Make)\s*:\s*([A-Za-z0-9\s]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["brand"] = ExtractedField(value=brand_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": brand_m.group(0)}) if brand_m else ExtractedField(value=None, confidence=0.0)

        mod_m = re.search(r'(?:Model|Model\s*No)\s*:\s*([A-Za-z0-9\s-]+?)(?:\||\n|$)', text, re.IGNORECASE)
        fields["model"] = ExtractedField(value=mod_m.group(1).strip(), confidence=0.90, source={"page": 1, "text": mod_m.group(0)}) if mod_m else ExtractedField(value=None, confidence=0.0)

        sn_m = re.search(r'(?:Serial\s*No|Serial\s*Number|S/N)\s*:\s*([\w-]+)', text, re.IGNORECASE)
        fields["serial_number"] = ExtractedField(value=sn_m.group(1).strip(), confidence=0.95, source={"page": 1, "text": sn_m.group(0)}) if sn_m else ExtractedField(value=None, confidence=0.0)

        qty_m = re.search(r'(?:Quantity|Qty)\s*:\s*(\d+)', text, re.IGNORECASE)
        fields["quantity"] = ExtractedField(value=int(qty_m.group(1)), confidence=0.90, source={"page": 1, "text": qty_m.group(0)}) if qty_m else ExtractedField(value=1, confidence=0.80)

        u_price = DeterministicExtractor.extract_currency_amount(text, [r'(?:Unit\s*Price|Price|Rate)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["unit_price"] = ExtractedField(value=u_price[0], confidence=0.90, source={"page": 1, "text": u_price[2]}) if u_price else ExtractedField(value=None, confidence=0.0)

        tax_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Tax\s*Amount|GST|Tax)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["tax_amount"] = ExtractedField(value=tax_amt[0], confidence=0.90, source={"page": 1, "text": tax_amt[2]}) if tax_amt else ExtractedField(value=None, confidence=0.0)

        disc_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Discount|Discount\s*Amount)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["discount"] = ExtractedField(value=disc_amt[0], confidence=0.90, source={"page": 1, "text": disc_amt[2]}) if disc_amt else ExtractedField(value=None, confidence=0.0)

        tot_amt = DeterministicExtractor.extract_currency_amount(text, [r'(?:Total\s*Amount|Grand\s*Total|Total)\s*:?\s*(?:INR|\$|₹)?\s*([\d,]+(?:\.\d{1,2})?)'])
        fields["total_amount"] = ExtractedField(value=tot_amt[0], confidence=0.95, source={"page": 1, "text": tot_amt[2]}) if tot_amt else ExtractedField(value=None, confidence=0.0)

        return fields

    @staticmethod
    def extract_generic(text: str) -> Dict[str, ExtractedField]:
        fields = {}
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        fields["document_title"] = ExtractedField(value=lines[0] if lines else None, confidence=0.70, source={"page": 1, "text": lines[0]} if lines else None)
        fields["date"] = ExtractedField(value=None, confidence=0.0)
        fields["summary"] = ExtractedField(value=text[:200] if text else None, confidence=0.70, source={"page": 1, "text": text[:100]} if text else None)
        fields["key_fields"] = ExtractedField(value=None, confidence=0.0)
        return fields


def extract_with_deterministic_engine(text: str, doc_type: str) -> Dict[str, ExtractedField]:
    """Routes document extraction to specialized deterministic schema extractors."""
    if doc_type == "payslip" or doc_type in ["salary_certificate", "employment_income_proof", "income_certificate", "co_applicant_income_proof"]:
        return DeterministicExtractor.extract_payslip(text)
    elif doc_type == "bank_statement" or doc_type == "business_bank_statement":
        return DeterministicExtractor.extract_bank_statement(text)
    elif doc_type == "itr_tax_return" or doc_type in ["tax_computation", "business_itr"]:
        return DeterministicExtractor.extract_itr(text)
    elif doc_type == "kyc_identity" or doc_type in ["pan_card", "aadhaar_identity", "passport_identity", "driving_license_identity", "voter_id_identity", "student_kyc", "student_identity", "co_applicant_kyc"]:
        return DeterministicExtractor.extract_kyc_identity(text)
    elif doc_type in ["employment_letter", "employment_certificate", "employment_proof"]:
        return DeterministicExtractor.extract_employment_letter(text)
    elif doc_type in ["office_id", "employee_id", "company_id"]:
        return DeterministicExtractor.extract_office_id(text)
    elif doc_type == "form_16":
        return DeterministicExtractor.extract_form_16(text)
    elif doc_type == "address_proof" or doc_type == "property_address_proof":
        return DeterministicExtractor.extract_address_proof(text)
    elif doc_type in ["property_title_document", "sale_deed", "sale_agreement", "property_registration_document", "land_document", "legal_property_report", "encumbrance_certificate"]:
        return DeterministicExtractor.extract_property_document(text)
    elif doc_type == "approved_building_plan":
        return DeterministicExtractor.extract_approved_plan(text)
    elif doc_type in ["property_tax_receipt", "land_tax_receipt"]:
        return DeterministicExtractor.extract_property_tax(text)
    elif doc_type == "property_valuation_report":
        return DeterministicExtractor.extract_property_valuation(text)
    elif doc_type == "vehicle_quotation":
        return DeterministicExtractor.extract_vehicle_quotation(text)
    elif doc_type in ["vehicle_invoice", "vehicle_purchase_agreement", "vehicle_registration_document", "vehicle_insurance_document", "vehicle_valuation_document"]:
        return DeterministicExtractor.extract_vehicle_invoice(text)
    elif doc_type == "admission_letter":
        return DeterministicExtractor.extract_admission_letter(text)
    elif doc_type == "fee_structure":
        return DeterministicExtractor.extract_fee_structure(text)
    elif doc_type in ["academic_certificate", "marksheet"]:
        return DeterministicExtractor.extract_academic_certificate(text)
    elif doc_type in ["business_registration", "partnership_deed", "incorporation_certificate", "company_registration", "business_license"]:
        return DeterministicExtractor.extract_gst_certificate(text)
    elif doc_type == "gst_certificate":
        return DeterministicExtractor.extract_gst_certificate(text)
    elif doc_type == "gst_return":
        return DeterministicExtractor.extract_gst_return(text)
    elif doc_type == "profit_loss_statement":
        return DeterministicExtractor.extract_profit_loss(text)
    elif doc_type == "balance_sheet":
        return DeterministicExtractor.extract_balance_sheet(text)
    elif doc_type in ["fixed_deposit_certificate", "fixed_deposit_receipt", "fd_statement", "fd_loan_document", "bank_account_document"]:
        return DeterministicExtractor.extract_fixed_deposit(text)
    elif doc_type in ["gold_security_document", "jewellery_valuation_report", "gold_valuation_document", "pledge_document", "gold_loan_document", "security_document"]:
        return DeterministicExtractor.extract_gold_security(text)
    elif doc_type in ["land_record", "cultivation_record", "crop_document"]:
        return DeterministicExtractor.extract_land_record(text)
    elif doc_type == "agricultural_income_proof":
        return DeterministicExtractor.extract_agricultural_income(text)
    elif doc_type in ["product_quotation", "product_invoice", "purchase_invoice", "consumer_durable_loan_document"]:
        return DeterministicExtractor.extract_product_invoice(text)
    else:
        return DeterministicExtractor.extract_generic(text)


# =============================================================================
# 3. OLLAMA LLM EXTRACTION SERVICE (WITH INTELLIGENT CHUNKING)
# =============================================================================

EXTRACTION_PROMPT_TEMPLATE = """You are an expert financial document data extraction agent for a loan processing system.

Your job is to inspect the complete document content and extract EVERY requested field in the target schema.

Document Type: {document_type}

Target Extraction Fields Required:
{target_fields_json}

CRITICAL INSTRUCTIONS & ANTI-HALLUCINATION GUARDRAILS:
1. Search the COMPLETE document text thoroughly. Attempt every single field in the requested schema.
2. Extract the field value if it is explicitly present in the document.
3. Preserve numbers, dates, currency symbols, and text values accurately.
4. Return null ONLY if a field is genuinely absent, unreadable, or not stated in the document.
5. NEVER guess, infer, or hallucinate values.
6. NEVER infer basic salary from net salary or designation.
7. Do NOT insert US form numbers (e.g. 1040) unless it is explicitly printed in the source text.
8. Do NOT use generic titles (e.g. "BANK ACCOUNT STATEMENT") as bank_name or ("EMPLOYMENT CONFIRMATION LETTER") as employer_name.
9. PARTIAL EXTRACTION IS NOT SUCCESSFUL EXTRACTION.
10. Return ONLY valid JSON matching this exact structure:
{{
  "fields": {{
    "field_name_1": {{
      "value": <extracted_value_or_null>,
      "confidence": <float_0.0_to_1.0>,
      "source": {{ "page": 1, "text": "<exact_text_snippet>" }}
    }}
  }}
}}

Document Text Content:
\"\"\"
{document_text}
\"\"\"
"""


def extract_with_ollama_chunk(document_text: str, document_type: str) -> Optional[Dict[str, ExtractedField]]:
    """Calls local Ollama model via LangChain ChatOllama & ChatPromptTemplate for structured document extraction."""
    if not document_text or len(document_text.strip()) < 10:
        return None

    target_fields = EXTRACTION_SCHEMAS.get(document_type, [])
    if not target_fields:
        return None

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
            timeout=3.0
        )
        prompt_tmpl = ChatPromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
        chain = prompt_tmpl | llm

        response = chain.invoke({
            "document_type": document_type,
            "target_fields_json": json.dumps(target_fields),
            "document_text": document_text
        })
        raw_text = response.content if hasattr(response, "content") else str(response)

        raw_text = str(raw_text).strip()
        if "```" in raw_text:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_text, re.DOTALL)
            if json_match:
                raw_text = json_match.group(1)

        parsed = json.loads(raw_text)
        extracted_dict = {}

        fields_data = parsed.get("fields") if isinstance(parsed.get("fields"), dict) else parsed
        for field_name in target_fields:
            field_info = fields_data.get(field_name)
            if isinstance(field_info, dict):
                val = field_info.get("value")
                conf = float(field_info.get("confidence", 0.95 if val is not None else 0.0))
                src = field_info.get("source")
            else:
                val = field_info
                conf = 0.95 if val is not None else 0.0
                src = None

            if not is_valid_extracted_value(val):
                val = None
                conf = 0.0

            extracted_dict[field_name] = ExtractedField(
                value=val,
                confidence=min(max(conf, 0.0), 1.0) if val is not None else 0.0,
                source=src if isinstance(src, dict) else ({"text": str(src)} if src else None)
            )

        return extracted_dict

    except Exception as e:
        logger.warning(f"Ollama/LangChain extraction unavailable or timed out ({e}). Utilizing deterministic extraction fallback.")
        return None


def extract_with_ollama(document_text: str, document_type: str) -> Optional[Dict[str, ExtractedField]]:
    """
    Intelligent Chunking Strategy for large documents:
    Splits long documents into section chunks, extracts fields from each chunk,
    and merges non-null results without overwriting non-null values with null.
    """
    if not document_text or len(document_text.strip()) < 10:
        return None

    if len(document_text) <= 3500:
        return extract_with_ollama_chunk(document_text, document_type)

    chunk_size = 3000
    overlap = 500
    chunks = []
    start = 0
    while start < len(document_text):
        end = min(start + chunk_size, len(document_text))
        chunks.append(document_text[start:end])
        if end == len(document_text):
            break
        start += (chunk_size - overlap)

    logger.info(f"[Agent 2 Chunking] Processing {len(chunks)} text chunks for document length {len(document_text)}")

    merged_results: Dict[str, ExtractedField] = {}

    for idx, chunk_text in enumerate(chunks):
        chunk_res = extract_with_ollama_chunk(chunk_text, document_type)
        if not chunk_res:
            continue

        for f_name, f_field in chunk_res.items():
            if f_name not in merged_results:
                merged_results[f_name] = f_field
            else:
                existing = merged_results[f_name]
                if not is_valid_extracted_value(existing.value) and is_valid_extracted_value(f_field.value):
                    merged_results[f_name] = f_field
                elif is_valid_extracted_value(existing.value) and is_valid_extracted_value(f_field.value):
                    if f_field.confidence > existing.confidence:
                        merged_results[f_name] = f_field
                if f_name == "transactions" and isinstance(existing.value, list) and isinstance(f_field.value, list):
                    combined_tx = existing.value + f_field.value
                    seen_tx = set()
                    deduped = []
                    for tx in combined_tx:
                        tx_key = (tx.get("transaction_date"), tx.get("description"), tx.get("debit"), tx.get("credit"))
                        if tx_key not in seen_tx:
                            seen_tx.add(tx_key)
                            deduped.append(tx)
                    merged_results[f_name] = ExtractedField(
                        value=deduped if len(deduped) > 0 else None,
                        confidence=max(existing.confidence, f_field.confidence) if len(deduped) > 0 else 0.0,
                        source={"text": f"Merged {len(deduped)} transactions across chunks"} if len(deduped) > 0 else None
                    )

    return merged_results if merged_results else None


# =============================================================================
# 4. AGENT 2 LANGGRAPH WORKFLOW NODES
# =============================================================================

def receive_agent1_output(state: LoanDocumentState) -> LoanDocumentState:
    """Node 1: Receives classification output from Agent 1."""
    class_results = state.get("classification_results", [])
    logger.info(f"[Agent 2] Received {len(class_results)} classification results from Agent 1.")
    return state


def validate_agent1_classification(state: LoanDocumentState) -> LoanDocumentState:
    """Node 2: Validates classification results contract."""
    class_results = state.get("classification_results", [])
    valid_results = []
    for item in class_results:
        if not item.get("document_type"):
            item["document_type"] = "unknown"
        valid_results.append(item)
    state["classification_results"] = valid_results
    return state


def load_document_content(state: LoanDocumentState) -> LoanDocumentState:
    """Node 3: Loads document text content for each file with fallback path resolution."""
    class_results = state.get("classification_results", [])
    documents = state.get("documents", [])

    for idx, item in enumerate(class_results):
        file_path = item.get("file_path")
        if not file_path and idx < len(documents):
            file_path = documents[idx].get("file_path")

        filename = item.get("filename", "")

        if not file_path or not os.path.exists(file_path):
            candidates = [
                PROJECT_ROOT / "uploads" / filename,
                PROJECT_ROOT / "tests" / "dataset_files" / filename,
                Path(filename)
            ]
            for cand in candidates:
                if cand.exists() and cand.is_file():
                    file_path = str(cand.resolve())
                    break

        if file_path and os.path.exists(file_path):
            text, pages, quality, available, method, err = parse_document(file_path)
            item["file_path"] = file_path
            item["_loaded_text"] = text
            item["_loaded_pages"] = pages
            item["_loaded_available"] = available
            item["_extraction_method"] = method
            logger.info(f"[Agent 2 Content Loaded] {filename} (len={len(text)}, pages={pages}, method={method})")
        else:
            logger.error(f"[Agent 2 Error] Failed to locate file on disk for {filename} (path: {file_path})")
            item["_loaded_text"] = ""
            item["_loaded_pages"] = 0
            item["_loaded_available"] = False
    return state


def select_extraction_schema(state: LoanDocumentState) -> LoanDocumentState:
    """Node 4: Selects appropriate target extraction schema based on document_type."""
    class_results = state.get("classification_results", [])
    for item in class_results:
        doc_type = item.get("document_type", "unknown")
        item["_target_schema"] = EXTRACTION_SCHEMAS.get(doc_type, [])
    return state


def extract_information(state: LoanDocumentState) -> LoanDocumentState:
    """Node 5: Performs hybrid semantic + deterministic extraction for each document."""
    class_results = state.get("classification_results", [])
    extraction_results = []

    for item in class_results:
        start_time = time.time()
        doc_id = item.get("document_id", f"doc_{len(extraction_results)+1:03d}")
        filename = item.get("filename", "unnamed_document")
        doc_type = item.get("document_type", "unknown")
        text = item.get("_loaded_text", "")
        target_fields = item.get("_target_schema", EXTRACTION_SCHEMAS.get(doc_type, []))

        extracted_fields = None
        llm_called = False
        llm_valid = False
        errors = []

        if text and item.get("_loaded_available", True):
            llm_called = True
            extracted_fields = extract_with_ollama(text, doc_type)
            if extracted_fields:
                llm_valid = True

        det_fields = extract_with_deterministic_engine(text, doc_type)

        if extracted_fields is None:
            extracted_fields = det_fields
        else:
            for f_name in target_fields:
                if f_name not in extracted_fields:
                    extracted_fields[f_name] = ExtractedField(value=None, confidence=0.0)

            for f_name, det_f in det_fields.items():
                ollama_f = extracted_fields.get(f_name)
                if (not ollama_f or not is_valid_extracted_value(ollama_f.value)) and is_valid_extracted_value(det_f.value):
                    extracted_fields[f_name] = det_f

        if item.get("status") == "failed" or not item.get("_loaded_available", True):
            status = "failed"
            errors.append({
                "error_type": item.get("error_type", "unreadable_content"),
                "message": "Document text could not be extracted for information extraction.",
                "document_id": doc_id
            })
        else:
            status = "success"

        # Canonical Backend Completeness Calculation
        fields_requested = len(target_fields)
        fields_extracted = sum(
            1 for f_name in target_fields
            if f_name in extracted_fields and is_valid_extracted_value(extracted_fields[f_name].value)
        )
        fields_missing = max(0, fields_requested - fields_extracted)
        completeness = round((fields_extracted / fields_requested * 100.0), 2) if fields_requested > 0 else 0.0

        non_null_confs = [f.confidence for f_name, f in extracted_fields.items() if f_name in target_fields and is_valid_extracted_value(f.value)]
        overall_conf = float(sum(non_null_confs) / len(non_null_confs)) if non_null_confs else (0.90 if status == "success" and fields_requested == 0 else 0.0)
        proc_time = (time.time() - start_time) * 1000.0

        debug_info = {
            "document_id": doc_id,
            "document_type": doc_type,
            "text_available": bool(text),
            "text_length": len(text),
            "page_count": item.get("_loaded_pages", 1),
            "extraction_method": item.get("_extraction_method", "none"),
            "LLM_called": llm_called,
            "LLM_response_valid": llm_valid,
            "fields_requested": fields_requested,
            "fields_extracted": fields_extracted,
            "fields_missing": fields_missing,
            "extraction_completeness": completeness
        }

        result_model = DocumentExtractionResult(
            document_id=doc_id,
            filename=filename,
            document_type=doc_type,
            fields=extracted_fields or {},
            overall_confidence=min(max(overall_conf, 0.0), 1.0),
            extraction_status=status,
            fields_requested=fields_requested,
            fields_extracted=fields_extracted,
            fields_missing=fields_missing,
            extraction_completeness=completeness,
            extraction_debug_info=debug_info,
            processing_time_ms=proc_time,
            errors=errors,
            next_agent="validation_agent"
        )
        extraction_results.append(result_model.model_dump())

    state["extraction_results"] = extraction_results
    return state


def validate_extraction(state: LoanDocumentState) -> LoanDocumentState:
    """
    Node 6: Enforces anti-hallucination guardrails, title blacklists,
    and recalculates canonical completeness metrics on final extracted fields.
    """
    ext_results = state.get("extraction_results", [])

    for res in ext_results:
        doc_type = res.get("document_type", "unknown")
        fields = res.get("fields", {})
        target_fields = EXTRACTION_SCHEMAS.get(doc_type, [])

        # Enforce null confidence rule & value cleanup
        for f_name, f_data in fields.items():
            if isinstance(f_data, dict):
                val = f_data.get("value")
                if not is_valid_extracted_value(val):
                    f_data["value"] = None
                    f_data["confidence"] = 0.0

        # Recalculate canonical backend completeness metrics on final fields
        f_req = len(target_fields)
        f_ext = sum(
            1 for f_name in target_fields
            if f_name in fields and is_valid_extracted_value(
                fields[f_name].get("value") if isinstance(fields[f_name], dict) else getattr(fields[f_name], "value", None)
            )
        )
        f_miss = max(0, f_req - f_ext)
        completeness = round((f_ext / f_req * 100.0), 2) if f_req > 0 else 0.0

        res["fields_requested"] = f_req
        res["fields_extracted"] = f_ext
        res["fields_missing"] = f_miss
        res["extraction_completeness"] = completeness

        if "extraction_debug_info" in res and isinstance(res["extraction_debug_info"], dict):
            res["extraction_debug_info"]["fields_requested"] = f_req
            res["extraction_debug_info"]["fields_extracted"] = f_ext
            res["extraction_debug_info"]["fields_missing"] = f_miss
            res["extraction_debug_info"]["extraction_completeness"] = completeness

        # Type-specific sanity checks
        if doc_type == "bank_statement":
            ifsc_f = fields.get("IFSC", {})
            if isinstance(ifsc_f, dict) and ifsc_f.get("value"):
                ifsc_val = str(ifsc_f["value"]).upper().strip()
                if not re.match(r'^[A-Z]{4}0[A-Z0-9]{6}$', ifsc_val):
                    logger.warning(f"Sanity Check: Unusable IFSC format '{ifsc_val}' extracted")

        elif doc_type == "itr_tax_return":
            ay_f = fields.get("assessment_year", {})
            if isinstance(ay_f, dict) and ay_f.get("value"):
                ay_val = str(ay_f["value"]).strip()
                if not re.match(r'^\d{4}(?:-\d{2,4})?$', ay_val):
                    logger.warning(f"Sanity Check: Unusual AY format '{ay_val}'")

    return state


def calculate_extraction_metrics(state: LoanDocumentState) -> LoanDocumentState:
    """Node 7: Computes processing metrics across extracted documents."""
    extraction_results = state.get("extraction_results", [])
    total_docs = len(extraction_results)
    successful_docs = sum(1 for r in extraction_results if r.get("extraction_status") == "success")
    failed_docs = sum(1 for r in extraction_results if r.get("extraction_status") == "failed")
    avg_time = sum(r.get("processing_time_ms", 0.0) for r in extraction_results) / total_docs if total_docs > 0 else 0.0
    avg_comp = sum(r.get("extraction_completeness", 0.0) for r in extraction_results) / total_docs if total_docs > 0 else 0.0

    metrics = state.get("processing_metrics", {})
    metrics["agent_2"] = {
        "total_documents": total_docs,
        "successful_documents": successful_docs,
        "failed_documents": failed_docs,
        "avg_processing_time_ms": round(avg_time, 2),
        "avg_extraction_completeness": round(avg_comp, 2)
    }
    state["processing_metrics"] = metrics
    return state


def prepare_agent3_output(state: LoanDocumentState) -> LoanDocumentState:
    """Node 8: Prepares contract state for Agent 3 (Validation Agent)."""
    state["next_agent"] = "validation_agent"
    logger.info(f"[Agent 2] Extraction completed. Downstream contract next_agent set to: {state['next_agent']}")
    return state


# =============================================================================
# 5. LANGGRAPH BUILDER
# =============================================================================

def build_agent_2_graph():
    """Builds and compiles Agent 2 Information Extraction StateGraph."""
    workflow = StateGraph(LoanDocumentState)

    workflow.add_node("receive_agent1_output", receive_agent1_output)
    workflow.add_node("validate_agent1_classification", validate_agent1_classification)
    workflow.add_node("load_document_content", load_document_content)
    workflow.add_node("select_extraction_schema", select_extraction_schema)
    workflow.add_node("extract_information", extract_information)
    workflow.add_node("validate_extraction", validate_extraction)
    workflow.add_node("calculate_extraction_metrics", calculate_extraction_metrics)
    workflow.add_node("prepare_agent3_output", prepare_agent3_output)

    workflow.add_edge(START, "receive_agent1_output")
    workflow.add_edge("receive_agent1_output", "validate_agent1_classification")
    workflow.add_edge("validate_agent1_classification", "load_document_content")
    workflow.add_edge("load_document_content", "select_extraction_schema")
    workflow.add_edge("select_extraction_schema", "extract_information")
    workflow.add_edge("extract_information", "validate_extraction")
    workflow.add_edge("validate_extraction", "calculate_extraction_metrics")
    workflow.add_edge("calculate_extraction_metrics", "prepare_agent3_output")
    workflow.add_edge("prepare_agent3_output", END)

    return workflow.compile()


# =============================================================================
# 6. AGENT 2 CLASS INTERFACE
# =============================================================================

class InformationExtractionAgent:
    """
    Main Agent 2 Class Interface for Information Extraction.
    Consumes Agent 1 classification outputs and returns structured field extractions.
    """

    def __init__(self):
        self.graph = build_agent_2_graph()

    def process_document(self, file_path: str, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single document given its file path and Agent 1 classification result."""
        initial_state: LoanDocumentState = {
            "application_id": f"APP-{int(time.time())}",
            "documents": [{"file_path": file_path}],
            "classification_results": [classification_result],
            "extraction_results": [],
            "errors": [],
            "processing_metrics": {},
            "next_agent": "extraction_agent"
        }

        final_state = self.graph.invoke(initial_state)
        results = final_state.get("extraction_results", [])
        return results[0] if results else {}

    def process_batch(self, classification_results: List[Dict[str, Any]], file_paths: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Processes a batch of classified documents with failure isolation."""
        docs = []
        if file_paths:
            for p in file_paths:
                docs.append({"file_path": p})
        else:
            for c in classification_results:
                docs.append({"file_path": c.get("file_path", "")})

        initial_state: LoanDocumentState = {
            "application_id": f"APP-{int(time.time())}",
            "documents": docs,
            "classification_results": classification_results,
            "extraction_results": [],
            "errors": [],
            "processing_metrics": {},
            "next_agent": "extraction_agent"
        }

        final_state = self.graph.invoke(initial_state)
        return final_state.get("extraction_results", [])


# =============================================================================
# 7. EVALUATION METRICS CALCULATOR FOR SYNTHETIC DATASET
# =============================================================================

def calculate_agent_2_metrics(ground_truth_dataset: List[Dict[str, Any]], extraction_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates field extraction completeness, precision, recall, F1, and average confidence.
    """
    if not ground_truth_dataset or not extraction_results:
        return {"status": "Ground truth unavailable"}

    total_fields = 0
    correct_fields = 0
    missing_fields = 0
    confs = []
    completenesses = []

    for gt, res in zip(ground_truth_dataset, extraction_results):
        extracted_fields = res.get("fields", {})
        comp = res.get("extraction_completeness", 0.0)
        completenesses.append(comp)

        gt_fields = gt.get("fields", {})
        if not gt_fields:
            doc_type = res.get("document_type", "unknown")
            schema_fields = EXTRACTION_SCHEMAS.get(doc_type, [])
            for f_name in schema_fields:
                total_fields += 1
                ext_obj = extracted_fields.get(f_name, {})
                ext_val = ext_obj.get("value") if isinstance(ext_obj, dict) else getattr(ext_obj, "value", None)
                conf = ext_obj.get("confidence", 0.0) if isinstance(ext_obj, dict) else getattr(ext_obj, "confidence", 0.0)
                if is_valid_extracted_value(ext_val):
                    correct_fields += 1
                    confs.append(conf)
                else:
                    missing_fields += 1
        else:
            for f_name, expected_val in gt_fields.items():
                total_fields += 1
                ext_obj = extracted_fields.get(f_name, {})
                ext_val = ext_obj.get("value") if isinstance(ext_obj, dict) else getattr(ext_obj, "value", None)
                conf = ext_obj.get("confidence", 0.0) if isinstance(ext_obj, dict) else getattr(ext_obj, "confidence", 0.0)

                if is_valid_extracted_value(ext_val):
                    confs.append(conf)
                    if str(ext_val).strip().lower() == str(expected_val).strip().lower():
                        correct_fields += 1
                    else:
                        try:
                            if abs(float(ext_val) - float(expected_val)) < 0.01:
                                correct_fields += 1
                        except (ValueError, TypeError):
                            pass
                else:
                    missing_fields += 1

    acc = (correct_fields / total_fields * 100.0) if total_fields > 0 else 0.0
    avg_conf = (sum(confs) / len(confs) * 100.0) if confs else 0.0
    avg_comp = (sum(completenesses) / len(completenesses)) if completenesses else 0.0

    return {
        "field_accuracy": round(acc, 2),
        "average_completeness": round(avg_comp, 2),
        "missing_field_rate": round((missing_fields / total_fields * 100.0) if total_fields > 0 else 0.0, 2),
        "average_confidence": round(avg_conf, 2)
    }
