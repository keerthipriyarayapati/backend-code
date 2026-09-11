"""
===============================================================================
AGENT 4 — CROSS-DOCUMENT VERIFICATION AGENT
===============================================================================
Single Python Implementation File for Agent 4.

Core Responsibility:
Perform cross-document consistency verification across multiple extracted loan documents:
1. Applicant Identity & Name Consistency (KYC vs Payslip vs Bank Statement vs Tax Return vs Title Deed, etc.)
2. Strict Identifiers Verification (PAN, GSTIN, Bank Account, IFSC, FD Number, Chassis, Engine Number, etc.)
3. Address Consistency & Normalization (KYC vs Bank Statement vs Employment Letter vs Title Deed)
4. Income & Salary Equivalence (Monthly Payslip * 12 vs Annual Form 16 / ITR vs Bank Salary Credit)
5. Employer Name Consistency (Payslip vs Employment Letter vs Form 16 vs Bank Statement)
6. Role-Based Verification & Separation (Student vs Co-applicant in Education Loans, Farmer vs Land Owner in Agri Loans)
7. Deterministic Missing Value Isolation (Missing/null fields evaluate strictly to UNVERIFIABLE, never MISMATCH)
8. High-Precision Normalization Engine (Names, Addresses, Dates, Currency, Strict Identifiers)
9. Semantic LLM Fallback (Ollama qwen2.5:7b for ambiguous semantic text comparisons)
10. Deterministic Consistency & Verifiability Coverage Scoring (Distinguishes consistency from verifiability)

Architecture & Downstream Contract:
- LangGraph StateGraph orchestration pipeline
- Inputs: List of Agent 2 Extraction Results OR Agent 3 Validation Results, Loan Type, Application ID
- Output: Structured CrossDocumentResult Pydantic model / Dict
- Downstream Contract: next_agent = "risk_anomaly_agent"
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
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

# Import project root and shared state
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.state import (
    CrossDocumentFinding,
    CrossDocumentResult,
    LoanDocumentState
)
from shared.policy import (
    get_loan_type_policy,
    LOAN_TYPE_NAMES
)

# Optional LangChain Ollama import
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
logger = logging.getLogger("Agent4_CrossDocumentAgent")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


# =============================================================================
# 1. HELPER FUNCTIONS & PII MASKING
# =============================================================================

def is_valid_value(val: Any) -> bool:
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
        cleaned = val.strip().lower()
        if not cleaned or cleaned in ["null", "none", "n/a", "na", "undefined", "unknown"]:
            return False
        return True
    if isinstance(val, (list, dict)):
        return len(val) > 0
    return bool(val)


def mask_pii(text: Any, field_type: str = "general") -> str:
    """
    Masks sensitive personally identifiable information (PII) for safe logging and display.
    - PAN (10 chars): e.g. ABCDE1234F -> ABCDE****F
    - Account / Card / Aadhaar: e.g. 123456789012 -> 1234****9012
    - General: masks inner characters if long enough
    """
    if not is_valid_value(text):
        return "N/A"
    
    val_str = str(text).strip()
    ftype = field_type.lower()
    
    if "pan" in ftype or (len(val_str) == 10 and val_str[:5].isalpha() and val_str[5:9].isdigit()):
        return f"{val_str[:5]}****{val_str[-1:]}"
        
    if "account" in ftype or "card" in ftype or "aadhaar" in ftype or val_str.isdigit():
        if len(val_str) > 6:
            return f"{val_str[:4]}****{val_str[-4:]}"
        return f"{val_str[:2]}****"
        
    return val_str


# =============================================================================
# 2. CANONICAL FIELD MAPPING MATRIX (AGENT 2 SCHEMAS ALIGNMENT)
# =============================================================================

CANONICAL_FIELD_MAPPINGS: Dict[str, Dict[str, List[str]]] = {
    "applicant_name": {
        "kyc_identity": ["full_name", "applicant_name", "name"],
        "student_kyc": ["full_name", "student_name", "applicant_name"],
        "co_applicant_kyc": ["full_name", "co_applicant_name", "applicant_name"],
        "pan_card": ["name", "full_name", "applicant_name"],
        "aadhaar_identity": ["full_name", "name"],
        "passport_identity": ["full_name", "name"],
        "voter_id_identity": ["full_name", "name"],
        "driving_license_identity": ["full_name", "name"],
        "payslip": ["employee_name", "applicant_name"],
        "salary_certificate": ["employee_name", "applicant_name"],
        "employment_letter": ["employee_name", "applicant_name"],
        "employment_certificate": ["employee_name", "applicant_name"],
        "office_id": ["employee_name", "applicant_name", "cardholder_name"],
        "employee_id": ["employee_name", "applicant_name", "cardholder_name"],
        "company_id": ["employee_name", "applicant_name", "cardholder_name"],
        "employment_proof": ["employee_name", "applicant_name"],
        "student_kyc": ["full_name", "student_name", "applicant_name"],
        "co_applicant_kyc": ["full_name", "co_applicant_name", "applicant_name"],
        "bank_statement": ["account_holder_name", "customer_name", "applicant_name"],
        "business_bank_statement": ["account_holder_name", "business_name"],
        "itr_tax_return": ["taxpayer_name", "applicant_name"],
        "form_16": ["employee_name", "applicant_name"],
        "property_title_document": ["owner_name", "buyer_name", "purchaser_name"],
        "sale_deed": ["owner_name", "buyer_name"],
        "sale_agreement": ["owner_name", "buyer_name"],
        "fixed_deposit_certificate": ["account_holder_name", "holder_name"],
        "fixed_deposit_receipt": ["account_holder_name", "holder_name"],
        "vehicle_quotation": ["customer_name", "buyer_name"],
        "vehicle_invoice": ["buyer_name", "customer_name"],
        "product_quotation": ["customer_name", "buyer_name"],
        "business_registration": ["owner_name", "promoter_name", "proprietor_name"],
        "gold_security_document": ["borrower_name", "applicant_name"],
        "jewellery_valuation_report": ["borrower_name", "applicant_name"],
        "land_record": ["farmer_name", "owner_name"],
        "cultivation_record": ["farmer_name", "applicant_name"],
        "agricultural_income_proof": ["farmer_name", "applicant_name"],
        "admission_letter": ["student_name", "applicant_name"],
        "fee_structure": ["student_name", "applicant_name"],
        "academic_certificate": ["student_name", "applicant_name"],
        "marksheet": ["student_name", "applicant_name"],
        "co_applicant_income_proof": ["employee_name", "applicant_name"]
    },
    "crop_type": {
        "cultivation_record": ["crop_type", "primary_crop"],
        "agricultural_income_proof": ["crop_type"]
    },
    "cultivated_area": {
        "cultivation_record": ["cultivated_area", "land_area"],
        "agricultural_income_proof": ["cultivated_area"]
    },
    "pan_number": {
        "pan_card": ["pan_number", "pan"],
        "kyc_identity": ["pan_number", "identity_document_number", "pan"],
        "payslip": ["employee_pan", "pan_number", "pan"],
        "salary_certificate": ["pan_number", "pan"],
        "bank_statement": ["pan_number", "pan"],
        "itr_tax_return": ["pan_number", "taxpayer_pan", "pan"],
        "form_16": ["employee_PAN", "pan_number", "pan"],
        "business_registration": ["pan_number", "pan"],
        "gst_certificate": ["PAN", "pan_number", "pan"],
        "fixed_deposit_certificate": ["pan_number", "pan"]
    },
    "address": {
        "kyc_identity": ["address"],
        "aadhaar_identity": ["address"],
        "passport_identity": ["address"],
        "address_proof": ["address"],
        "bank_statement": ["address"],
        "employment_letter": ["work_location", "address"],
        "itr_tax_return": ["address"],
        "property_title_document": ["property_address"],
        "business_registration": ["registered_address"],
        "gst_certificate": ["registered_address", "principal_place_of_business"]
    },
    "employer_name": {
        "payslip": ["employer_name", "company_name"],
        "salary_certificate": ["employer_name", "company_name"],
        "employment_letter": ["employer_name", "company_name"],
        "employment_certificate": ["employer_name", "company_name"],
        "office_id": ["employer_name", "company_name"],
        "employee_id": ["employer_name", "company_name"],
        "company_id": ["employer_name", "company_name"],
        "employment_proof": ["employer_name", "company_name"],
        "form_16": ["employer_name", "company_name"],
        "bank_statement": ["employer_name", "salary_credit_description"]
    },
    "monthly_salary_income": {
        "payslip": ["gross_salary", "monthly_gross_salary", "basic_salary", "net_salary"],
        "salary_certificate": ["gross_salary", "monthly_salary", "annual_salary"],
        "employment_letter": ["salary_if_explicitly_stated", "annual_salary"],
        "form_16": ["salary_income", "gross_salary"],
        "itr_tax_return": ["gross_total_income", "taxable_income"]
    },
    "bank_account_number": {
        "bank_statement": ["account_number", "bank_account_number"],
        "payslip": ["bank_account_number", "account_number"],
        "salary_certificate": ["bank_account_number"],
        "fixed_deposit_certificate": ["account_number"]
    },
    "ifsc_code": {
        "bank_statement": ["IFSC", "ifsc_code", "ifsc"],
        "payslip": ["ifsc_code", "ifsc"],
        "fixed_deposit_certificate": ["ifsc_code", "ifsc"]
    },
    "fd_number": {
        "fixed_deposit_certificate": ["FD_number", "fd_number"],
        "fixed_deposit_receipt": ["FD_number", "fd_number"],
        "fd_statement": ["FD_number", "fd_number"],
        "bank_account_document": ["FD_number", "fd_number"]
    },
    "chassis_number": {
        "vehicle_quotation": ["chassis_number"],
        "vehicle_invoice": ["chassis_number"]
    },
    "engine_number": {
        "vehicle_quotation": ["engine_number"],
        "vehicle_invoice": ["engine_number"]
    },
    "gstin": {
        "gst_certificate": ["GSTIN", "gstin"],
        "gst_return": ["GSTIN", "gstin"],
        "business_registration": ["gstin", "GSTIN"]
    },
    "student_name": {
        "admission_letter": ["student_name"],
        "fee_structure": ["student_name"],
        "academic_certificate": ["student_name"],
        "marksheet": ["student_name"],
        "student_kyc": ["student_name", "full_name"]
    },
    "net_weight": {
        "gold_security_document": ["net_weight", "gross_weight"],
        "jewellery_valuation_report": ["net_weight", "gross_weight"]
    },
    "purity": {
        "gold_security_document": ["purity"],
        "jewellery_valuation_report": ["purity"]
    },
    "survey_number": {
        "property_title_document": ["survey_number", "plot_number"],
        "sale_deed": ["survey_number"],
        "sale_agreement": ["survey_number"],
        "land_record": ["survey_number"]
    }
}


def get_canonical_value(doc_obj: Dict[str, Any], canonical_field: str) -> Tuple[Optional[Any], Optional[str]]:
    """
    Safely resolves a canonical business field from an Agent 2/3 document result object.
    
    Handles:
    - ExtractedField objects (reads .value)
    - Dicts with "value" key
    - Direct primitive values
    - Document-type specific field aliases via CANONICAL_FIELD_MAPPINGS
    - Missing data strictly returns None (NEVER string "None" or "null")
    """
    if not doc_obj or not isinstance(doc_obj, dict):
        return None, None
        
    doc_type = doc_obj.get("document_type", "").lower().strip()
    
    # 1. Look up alias candidates for this canonical field and document type
    field_candidates = []
    if canonical_field in CANONICAL_FIELD_MAPPINGS:
        mapping = CANONICAL_FIELD_MAPPINGS[canonical_field]
        if doc_type in mapping:
            field_candidates.extend(mapping[doc_type])
            
    if canonical_field not in field_candidates:
        field_candidates.append(canonical_field)
        
    fields = doc_obj.get("fields", {})
    
    # Helper to extract value from any field wrapper
    def extract_val(field_entry: Any) -> Optional[Any]:
        if field_entry is None:
            return None
        if hasattr(field_entry, "value"):
            v = field_entry.value
        elif isinstance(field_entry, dict) and "value" in field_entry:
            v = field_entry["value"]
        else:
            v = field_entry
            
        if is_valid_value(v):
            return v
        return None

    # 2. Search in 'fields' map
    for cand in field_candidates:
        if cand in fields:
            val = extract_val(fields[cand])
            if val is not None:
                return val, cand

    # 3. Search top-level dict
    for cand in field_candidates:
        if cand in doc_obj and cand not in ["fields", "document_type", "document_id", "filename", "findings"]:
            val = extract_val(doc_obj[cand])
            if val is not None:
                return val, cand

    return None, None


# =============================================================================
# 3. DETERMINISTIC NORMALIZERS ENGINE
# =============================================================================

SALUTATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "shri", "smt", "kumari", "sir", "madam",
    "master", "miss", "late", "capt", "col", "maj"
}

STRICT_IDENTIFIER_FIELDS = {
    "pan_number", "pan", "gst_number", "gstin", "ifsc_code", "ifsc",
    "account_number", "bank_account_number", "fd_number", "fixed_deposit_number",
    "chassis_number", "engine_number", "vehicle_registration_number", "survey_number",
    "aadhaar_number"
}


def normalize_name(name_str: str) -> str:
    """
    Normalizes person/entity name by lowercasing, stripping salutations, removing punctuation,
    and standardizing whitespace.
    """
    if not is_valid_value(name_str):
        return ""
    
    s = str(name_str).lower().strip()
    s = re.sub(r'[^\w\s]', ' ', s)
    tokens = s.split()
    
    # Remove leading salutations
    cleaned_tokens = []
    for idx, token in enumerate(tokens):
        if idx == 0 and token in SALUTATIONS:
            continue
        if idx == 1 and cleaned_tokens == [] and token in SALUTATIONS:
            continue
        cleaned_tokens.append(token)
        
    return " ".join(cleaned_tokens)


def normalize_address(addr_str: str) -> str:
    """
    Normalizes street/residential address by expanding common abbreviations, lowercasing,
    and removing non-essential punctuation.
    """
    if not is_valid_value(addr_str):
        return ""
    
    s = str(addr_str).lower().strip()
    
    abbreviations = {
        r'\bapt\b': 'apartment',
        r'\bflat\b': 'apartment',
        r'\bst\b': 'street',
        r'\brd\b': 'road',
        r'\bave\b': 'avenue',
        r'\bblvd\b': 'boulevard',
        r'\bno\b': 'number',
        r'\bflr\b': 'floor',
        r'\bbldg\b': 'building',
        r'\bsoc\b': 'society',
        r'\bnr\b': 'near',
        r'\bopp\b': 'opposite',
        r'\bhyd\b': 'hyderabad',
        r'\bblr\b': 'bangalore',
        r'\bbengaluru\b': 'bangalore',
        r'\bchennai\b': 'chennai',
        r'\bmum\b': 'mumbai',
        r'\bdelhi\b': 'delhi',
        r'\bdist\b': 'district',
        r'\bpo\b': 'post office',
        r'\bps\b': 'police station'
    }
    
    for pattern, repl in abbreviations.items():
        s = re.sub(pattern, repl, s)
        
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def normalize_date(date_val: Any) -> Optional[str]:
    """
    Normalizes date into standard ISO YYYY-MM-DD string format.
    """
    if not is_valid_value(date_val):
        return None
        
    val_str = str(date_val).strip()
    
    # Try ISO YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
        return val_str
        
    # Try common formats
    formats = [
        "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
        "%Y/%m/%d", "%Y-%m-%d",
        "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y", "%d %B %Y"
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(val_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
            
    return val_str.lower()


def normalize_currency(val: Any) -> Optional[float]:
    """
    Normalizes monetary salary/income values into clean float.
    Strips currency symbols (₹, $, EUR), commas, words (per month, pa, etc.).
    """
    if not is_valid_value(val):
        return None
        
    if isinstance(val, (int, float)):
        return float(val)
        
    s = str(val).strip()
    s = re.sub(r'[₹\$€,\s]', '', s)
    s = re.sub(r'(?i)(inr|rs|\/month|\/year|pa|pm|per month|per annum)', '', s).strip()
    
    try:
        return float(s)
    except ValueError:
        return None


def normalize_identifier(id_str: str) -> str:
    """
    Normalizes strict alphanumeric identifiers (PAN, GST, IFSC, Account Number).
    """
    if not is_valid_value(id_str):
        return ""
    s = str(id_str).upper().strip()
    s = re.sub(r'[\s\-\.\/]', '', s)
    return s


# =============================================================================
# 4. FIELD COMPARISON LOGIC
# =============================================================================

def compare_names(name1: str, name2: str) -> Tuple[str, float, str]:
    """
    Compares two person/entity names using deterministic normalization & token analysis.
    Returns: (comparison_result, confidence, explanation)
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if norm1 == norm2:
        return ("MATCH", 1.0, f"Exact name match: '{name1}' equals '{name2}'")
        
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    
    # Exact token set match (e.g. "Ananya Rao" vs "Rao Ananya")
    if tokens1 == tokens2:
        return ("MATCH", 0.98, f"Name word order variation: '{name1}' vs '{name2}'")
        
    # Subset token match or initial match (e.g. "Ananya K. Rao" vs "Ananya Rao")
    if tokens1.issubset(tokens2) or tokens2.issubset(tokens1):
        return ("MINOR_VARIATION", 0.90, f"Minor name abbreviation/initial variation: '{name1}' vs '{name2}'")
        
    # First and last name token match
    list1 = norm1.split()
    list2 = norm2.split()
    if len(list1) >= 2 and len(list2) >= 2:
        if list1[0] == list2[0] and list1[-1] == list2[-1]:
            return ("MINOR_VARIATION", 0.88, f"Minor middle name variation: '{name1}' vs '{name2}'")
            
    # Fuzzy token overlap
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    if union:
        jaccard = len(intersection) / len(union)
        if jaccard >= 0.5:
            return ("MINOR_VARIATION", 0.75, f"Partial name overlap: '{name1}' vs '{name2}'")
            
    return ("MISMATCH", 0.0, f"Name mismatch: '{name1}' does not match '{name2}'")


def compare_addresses(addr1: str, addr2: str) -> Tuple[str, float, str]:
    """
    Compares two addresses using normalized token overlap analysis.
    """
    norm1 = normalize_address(addr1)
    norm2 = normalize_address(addr2)
    
    if norm1 == norm2:
        return ("MATCH", 1.0, "Exact address match")
        
    tokens1 = set(norm1.split())
    tokens2 = set(norm2.split())
    
    if not tokens1 or not tokens2:
        return ("UNVERIFIABLE", 0.0, "Insufficient address tokens to compare")
        
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    jaccard = len(intersection) / len(union)
    
    if jaccard >= 0.65:
        return ("MATCH", 0.95, f"Address match with minor formatting variations ({jaccard*100:.1f}% token overlap)")
    elif jaccard >= 0.40 or norm1 in norm2 or norm2 in norm1:
        return ("MINOR_VARIATION", 0.80, f"Minor address variation ({jaccard*100:.1f}% token overlap)")
    else:
        return ("MISMATCH", 0.0, f"Address mismatch: '{mask_pii(addr1, 'address')}' vs '{mask_pii(addr2, 'address')}'")


def compare_salaries(val1: float, type1: str, val2: float, type2: str) -> Tuple[str, float, str]:
    """
    Compares monthly vs annual salary figures with period equivalence normalization.
    """
    t1 = type1.lower()
    t2 = type2.lower()
    
    is_annual1 = any(kw in t1 for kw in ["annual", "itr", "tax_return", "form_16", "gross_total_income", "annual_salary"])
    is_annual2 = any(kw in t2 for kw in ["annual", "itr", "tax_return", "form_16", "gross_total_income", "annual_salary"])
    
    annual1 = val1 if is_annual1 else val1 * 12.0
    annual2 = val2 if is_annual2 else val2 * 12.0
    
    if annual1 <= 0 or annual2 <= 0:
        return ("UNVERIFIABLE", 0.0, "Zero or negative salary value cannot be verified")
        
    ratio = min(annual1, annual2) / max(annual1, annual2)
    
    desc1 = f"₹{val1:,.2f} ({'annual' if is_annual1 else 'monthly'})"
    desc2 = f"₹{val2:,.2f} ({'annual' if is_annual2 else 'monthly'})"
    
    if ratio >= 0.92:
        return ("MATCH", 0.95, f"Salary values match across documents: {desc1} vs {desc2}")
    elif ratio >= 0.80:
        return ("MINOR_VARIATION", 0.85, f"Salary values match within acceptable bonus/allowance variance: {desc1} vs {desc2}")
    else:
        return ("MISMATCH", 0.0, f"Significant salary mismatch: {desc1} vs {desc2} (ratio: {ratio:.2f})")


def normalize_survey_number(surv_str: str) -> str:
    """
    Normalizes survey numbers by upper-casing, removing spaces, hyphens, and slashes.
    Example: '142/3A' -> '1423A', '142 / 3A' -> '1423A', '142-3A' -> '1423A'
    """
    if not is_valid_value(surv_str):
        return ""
    s = str(surv_str).upper().strip()
    s = re.sub(r'[\s\-_]', '', s)
    return s


def normalize_area_acres(area_val: Any) -> Optional[float]:
    """
    Normalizes land/cultivated area values into acres.
    Converts hectares to acres if indicated.
    """
    if not is_valid_value(area_val):
        return None
    s = str(area_val).lower().strip()
    match = re.search(r'([\d\.]+)', s)
    if not match:
        return None
    try:
        val = float(match.group(1))
    except ValueError:
        return None
    if "hectare" in s or "ha" in s:
        return val * 2.47105  # Convert hectares to acres
    return val


def compare_strict_identifiers(id1: str, id2: str, field_name: str) -> Tuple[str, float, str]:
    """
    Compares strict alphanumeric identifiers (PAN, GSTIN, IFSC, Account Number, etc.).
    MISMATCH on strict identifiers yields CRITICAL severity.
    """
    norm1 = normalize_identifier(id1)
    norm2 = normalize_identifier(id2)
    
    masked1 = mask_pii(id1, field_name)
    masked2 = mask_pii(id2, field_name)
    
    if norm1 == norm2:
        return ("MATCH", 1.0, f"Exact match for identifier {field_name}: '{masked1}'")
    else:
        return ("MISMATCH", 0.0, f"Critical identifier mismatch for {field_name}: '{masked1}' vs '{masked2}'")


def compare_survey_numbers(s1: str, s2: str) -> Tuple[str, float, str]:
    """
    Compares two survey numbers with whitespace, hyphen, and slash normalization.
    """
    norm1 = normalize_survey_number(s1)
    norm2 = normalize_survey_number(s2)
    
    if not norm1 or not norm2:
        return ("UNVERIFIABLE", 0.0, f"Cannot compare survey numbers: '{s1}' vs '{s2}'")
        
    if norm1 == norm2:
        return ("MATCH", 1.0, f"Survey number match: '{s1}' equals '{s2}'")
    else:
        return ("MISMATCH", 0.0, f"Survey number mismatch: '{s1}' vs '{s2}'")


def compare_areas(a1: Any, a2: Any) -> Tuple[str, float, str]:
    """
    Compares land/cultivated area with unit normalization (acres/hectares).
    """
    val1 = normalize_area_acres(a1)
    val2 = normalize_area_acres(a2)
    
    if val1 is None or val2 is None or val1 <= 0 or val2 <= 0:
        return ("UNVERIFIABLE", 0.0, f"Cannot compare area values: '{a1}' vs '{a2}'")
        
    ratio = min(val1, val2) / max(val1, val2)
    if ratio >= 0.95:
        return ("MATCH", 1.0, f"Area match: '{a1}' equals '{a2}' ({val1:.2f} acres)")
    elif ratio >= 0.80:
        return ("MINOR_VARIATION", 0.85, f"Minor area variation: '{a1}' vs '{a2}' ({val1:.2f} vs {val2:.2f} acres)")
    else:
        return ("MISMATCH", 0.0, f"Area mismatch: '{a1}' vs '{a2}' ({val1:.2f} vs {val2:.2f} acres)")


def compare_land_owner_vs_farmer(name1: str, name2: str) -> Tuple[str, float, str]:
    """
    Compares Land Owner name vs Cultivator/Farmer name.
    In agriculture loans, tenant farmers may cultivate land owned by a different person.
    If names differ, return MINOR_VARIATION / POSSIBLE_MATCH instead of severe MISMATCH.
    """
    res, conf, msg = compare_names(name1, name2)
    if res == "MATCH":
        return ("MATCH", 1.0, f"Land owner and farmer/cultivator match: '{name1}'")
    elif res == "MINOR_VARIATION":
        return ("MINOR_VARIATION", 0.90, f"Minor land owner vs farmer name variation: '{name1}' vs '{name2}'")
    else:
        return ("MINOR_VARIATION", 0.70, f"Land owner ('{name1}') differs from farmer/cultivator ('{name2}') — acceptable tenant/family cultivation role.")


def compare_text_exact(t1: str, t2: str, field_name: str) -> Tuple[str, float, str]:
    """
    Compares text fields (e.g. crop_type).
    """
    if not is_valid_value(t1) or not is_valid_value(t2):
        return ("UNVERIFIABLE", 0.0, f"Cannot compare '{field_name}': value missing")
    n1 = str(t1).lower().strip()
    n2 = str(t2).lower().strip()
    if n1 == n2 or n1 in n2 or n2 in n1:
        return ("MATCH", 1.0, f"Matching {field_name}: '{t1}' equals '{t2}'")
    return ("MISMATCH", 0.0, f"Mismatch for {field_name}: '{t1}' vs '{t2}'")


# =============================================================================
# 5. LLM SEMANTIC FALLBACK (OPTIONAL)
# =============================================================================

def semantic_llm_compare(
    field_name: str,
    val1: Any,
    doc1_type: str,
    val2: Any,
    doc2_type: str
) -> Optional[Tuple[str, float, str]]:
    """
    Optional LLM semantic fallback via Ollama qwen2.5:7b for ambiguous text comparisons.
    LLM MUST NOT override strict identifier mismatches.
    """
    if not LANGCHAIN_AVAILABLE or field_name in STRICT_IDENTIFIER_FIELDS:
        return None
        
    try:
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=OLLAMA_MODEL,
            temperature=0.0,
            timeout=5.0
        )
        
        prompt = f"""You are a senior banking auditor verifying cross-document consistency for a loan application.
Compare the following two extracted field values:

Field Name: {field_name}
Document 1 ({doc1_type}): "{val1}"
Document 2 ({doc2_type}): "{val2}"

Analyze if these represent the same entity, person, company, or address with acceptable spelling or abbreviation variations.
Respond ONLY with a JSON object containing:
- "result": "MATCH" | "MINOR_VARIATION" | "MISMATCH"
- "confidence": float between 0.0 and 1.0
- "explanation": concise 1-sentence reason

JSON Response:"""

        response = llm.invoke([
            SystemMessage(content="You are a strict data verification model. Output valid JSON only."),
            HumanMessage(content=prompt)
        ])
        
        content = response.content.strip()
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            res = data.get("result", "MISMATCH").upper()
            conf = float(data.get("confidence", 0.8))
            expl = data.get("explanation", "Semantic LLM verification completed.")
            if res in ["MATCH", "MINOR_VARIATION", "MISMATCH"]:
                return (res, conf, f"Semantic LLM: {expl}")
    except Exception as e:
        logger.debug(f"Ollama LLM fallback skipped or failed: {str(e)}")
        
    return None


# =============================================================================
# 6. CROSS DOCUMENT COMPARISON POLICY MATRIX
# =============================================================================

# Rule definitions per document type pair
CROSS_DOCUMENT_FIELD_MAPPINGS = [
    # Applicant Name Rules
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card", "aadhaar_identity", "passport_identity"],
        "doc2_types": ["payslip", "salary_certificate", "employment_letter"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card"],
        "doc2_types": ["bank_statement", "business_bank_statement"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card"],
        "doc2_types": ["itr_tax_return", "form_16"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card"],
        "doc2_types": ["property_title_document", "sale_deed", "sale_agreement"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card"],
        "doc2_types": ["fixed_deposit_certificate", "fixed_deposit_receipt"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card"],
        "doc2_types": ["vehicle_quotation", "vehicle_invoice", "product_quotation"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },

    # Strict PAN Number Rules (CRITICAL)
    {
        "field_name": "pan_number",
        "doc1_types": ["pan_card", "kyc_identity"],
        "doc2_types": ["itr_tax_return", "form_16", "bank_statement", "business_registration", "gst_certificate", "fixed_deposit_certificate"],
        "comparison_type": "strict_identifier",
        "severity_on_mismatch": "CRITICAL"
    },

    # Address Rules
    {
        "field_name": "address",
        "doc1_types": ["kyc_identity", "address_proof", "aadhaar_identity", "passport_identity"],
        "doc2_types": ["bank_statement", "employment_letter", "itr_tax_return"],
        "comparison_type": "address",
        "severity_on_mismatch": "MEDIUM"
    },

    # Employer Name Rules
    {
        "field_name": "employer_name",
        "doc1_types": ["payslip", "salary_certificate"],
        "doc2_types": ["employment_letter", "form_16"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },

    # Income / Salary Rules
    {
        "field_name": "monthly_salary_income",
        "doc1_types": ["payslip", "salary_certificate"],
        "doc2_types": ["itr_tax_return", "form_16"],
        "comparison_type": "salary",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "salary_credit_bank",
        "doc1_types": ["payslip", "salary_certificate"],
        "doc2_types": ["bank_statement"],
        "comparison_type": "salary",
        "severity_on_mismatch": "HIGH"
    },

    # Strict Bank Account Rules (CRITICAL)
    {
        "field_name": "bank_account_number",
        "doc1_types": ["payslip", "salary_certificate"],
        "doc2_types": ["bank_statement"],
        "comparison_type": "strict_identifier",
        "severity_on_mismatch": "CRITICAL"
    },

    # Education Loan Specific Rules (Role-Aware)
    {
        "field_name": "applicant_name",
        "doc1_types": ["student_kyc", "kyc_identity", "aadhaar_identity", "passport_identity"],
        "doc2_types": ["admission_letter", "fee_structure", "academic_certificate", "marksheet"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["student_kyc", "kyc_identity", "aadhaar_identity", "passport_identity"],
        "doc2_types": ["co_applicant_income_proof", "co_applicant_kyc", "co_applicant_bank"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["admission_letter", "fee_structure"],
        "doc2_types": ["academic_certificate", "marksheet"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "institution_name",
        "doc1_types": ["admission_letter", "fee_structure"],
        "doc2_types": ["academic_certificate", "marksheet"],
        "comparison_type": "text",
        "severity_on_mismatch": "MEDIUM"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["co_applicant_kyc", "pan_card", "kyc_identity"],
        "doc2_types": ["co_applicant_income_proof", "payslip", "salary_certificate", "itr_tax_return"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    # Agriculture Specific Rules
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card", "aadhaar_identity"],
        "doc2_types": ["land_record", "land_ownership_document"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card", "aadhaar_identity"],
        "doc2_types": ["cultivation_record", "crop_document"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card", "aadhaar_identity"],
        "doc2_types": ["agricultural_income_proof"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["land_record", "land_ownership_document"],
        "doc2_types": ["cultivation_record", "crop_document"],
        "comparison_type": "land_owner_vs_farmer",
        "severity_on_mismatch": "LOW"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["cultivation_record", "crop_document"],
        "doc2_types": ["agricultural_income_proof"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["agricultural_income_proof"],
        "doc2_types": ["bank_statement", "business_bank_statement"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "survey_number",
        "doc1_types": ["land_record", "land_ownership_document"],
        "doc2_types": ["cultivation_record", "crop_document"],
        "comparison_type": "survey_number",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "subdivision_number",
        "doc1_types": ["land_record", "land_ownership_document"],
        "doc2_types": ["cultivation_record", "crop_document"],
        "comparison_type": "strict_identifier",
        "severity_on_mismatch": "MEDIUM"
    },
    {
        "field_name": "village",
        "doc1_types": ["land_record", "land_ownership_document"],
        "doc2_types": ["cultivation_record", "crop_document"],
        "comparison_type": "address",
        "severity_on_mismatch": "MEDIUM"
    },
    {
        "field_name": "district",
        "doc1_types": ["land_record", "land_ownership_document"],
        "doc2_types": ["cultivation_record", "crop_document"],
        "comparison_type": "address",
        "severity_on_mismatch": "MEDIUM"
    },
    {
        "field_name": "crop_type",
        "doc1_types": ["cultivation_record", "crop_document"],
        "doc2_types": ["agricultural_income_proof"],
        "comparison_type": "text",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "cultivated_area",
        "doc1_types": ["cultivation_record", "crop_document"],
        "doc2_types": ["agricultural_income_proof"],
        "comparison_type": "area",
        "severity_on_mismatch": "MEDIUM"
    },

    # Property Specific Cross-Document Rules (Home Loan / Loan Against Property)
    {
        "field_name": "property_address",
        "doc1_types": ["property_title_document", "sale_deed"],
        "doc2_types": ["sale_agreement", "approved_building_plan", "property_tax_receipt", "property_valuation_report"],
        "comparison_type": "address",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "plot_number",
        "doc1_types": ["property_title_document", "sale_deed"],
        "doc2_types": ["sale_agreement", "approved_building_plan"],
        "comparison_type": "survey_number",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "survey_number",
        "doc1_types": ["property_title_document", "sale_deed"],
        "doc2_types": ["sale_agreement", "approved_building_plan"],
        "comparison_type": "survey_number",
        "severity_on_mismatch": "HIGH"
    },
    {
        "field_name": "applicant_name",
        "doc1_types": ["kyc_identity", "pan_card"],
        "doc2_types": ["approved_building_plan"],
        "comparison_type": "name",
        "severity_on_mismatch": "HIGH"
    }
]


def resolve_document_role(doc_obj: Dict[str, Any], loan_type: str = "") -> str:
    """
    Determines document entity role: STUDENT, CO_APPLICANT, FARMER, LAND_OWNER, or PRIMARY.
    Uses slot policy as authoritative source, then metadata and filename heuristics.
    """
    if not isinstance(doc_obj, dict):
        return "PRIMARY"

    explicit_role = doc_obj.get("entity_role") or doc_obj.get("document_role")
    if explicit_role:
        return str(explicit_role).upper()

    req_id = doc_obj.get("requirement_id") or doc_obj.get("slot_id")
    if req_id:
        policy = get_loan_type_policy(loan_type or "education_loan")
        for req in policy.get("required", []) + policy.get("optional", []):
            if req.requirement_id == req_id and hasattr(req, "entity_role") and req.entity_role:
                return req.entity_role.upper()

        if "co_app" in req_id or "coapplicant" in req_id:
            return "CO_APPLICANT"
        if "student" in req_id or "academic" in req_id or "admission" in req_id or "fee" in req_id:
            return "STUDENT"

    dtype = (doc_obj.get("document_type") or "").lower()
    fname = (doc_obj.get("filename") or doc_obj.get("file_name") or doc_obj.get("uploaded_filename") or "").lower()

    if "co_applicant" in dtype or "coapplicant" in dtype or "co_app" in dtype:
        return "CO_APPLICANT"
    if "student" in dtype:
        return "STUDENT"

    if any(k in fname for k in ["co_applicant", "coapplicant", "co-applicant", "coapp"]):
        return "CO_APPLICANT"
    if "student" in fname:
        return "STUDENT"

    if (loan_type or "").lower() == "education_loan":
        if dtype in ["admission_letter", "fee_structure", "academic_certificate", "marksheet", "student_kyc"]:
            return "STUDENT"
        if dtype in ["co_applicant_kyc", "co_applicant_income_proof", "co_applicant_bank"]:
            return "CO_APPLICANT"

    return "PRIMARY"


# =============================================================================
# 7. CORE CROSS-DOCUMENT VERIFIER CLASS
# =============================================================================

class CrossDocumentVerifier:
    """
    Core verification engine that compares extracted document data against policies
    and generates structured findings, verification coverage metrics, and consistency scores.
    """

    def process(
        self,
        document_results: List[Dict[str, Any]],
        loan_type: str = "personal_loan",
        application_id: str = "APP-GENERIC",
        min_coverage: float = 80.0
    ) -> CrossDocumentResult:
        """
        Executes full cross-document verification on list of extracted/validated document objects.
        """
        start_time = time.time()
        logger.info(f"Starting Cross-Document Verification for App: '{application_id}', Loan Type: '{loan_type}' across {len(document_results)} documents.")

        # Store raw document objects indexed by document_type
        docs_by_type: Dict[str, Dict[str, Any]] = {}
        for d in document_results:
            if not isinstance(d, dict):
                continue
            dtype = d.get("document_type", "unknown").lower().strip()
            docs_by_type[dtype] = d
            
            # Log extracted fields for transparency
            fields_dict = d.get("fields", {})
            non_null = [k for k, v in fields_dict.items() if is_valid_value(v.value if hasattr(v, "value") else (v.get("value") if isinstance(v, dict) else v))]
            logger.info(f"Agent 2 Doc Received: doc_id={d.get('document_id')}, type={dtype}, non_null_fields={len(non_null)}/{len(fields_dict)} ({non_null})")

        findings: List[CrossDocumentFinding] = []
        match_count = 0
        minor_var_count = 0
        mismatch_count = 0
        unverifiable_count = 0
        na_count = 0

        # Execute Rule Matrix Comparisons
        for rule in CROSS_DOCUMENT_FIELD_MAPPINGS:
            fname = rule["field_name"]
            d1_types = rule["doc1_types"]
            d2_types = rule["doc2_types"]
            comp_type = rule["comparison_type"]
            sev_mismatch = rule["severity_on_mismatch"]

            # Find present matching document objects
            matching_d1_type = next((t for t in d1_types if t in docs_by_type), None)
            matching_d2_type = next((t for t in d2_types if t in docs_by_type), None)

            if not matching_d1_type or not matching_d2_type:
                continue

            doc1_obj = docs_by_type[matching_d1_type]
            doc2_obj = docs_by_type[matching_d2_type]

            # Role Separation Check for Education Loan & Role-Aware Loans
            role1 = resolve_document_role(doc1_obj, loan_type)
            role2 = resolve_document_role(doc2_obj, loan_type)

            if loan_type.lower() == "education_loan" and role1 != role2:
                findings.append(CrossDocumentFinding(
                    field_name=fname,
                    doc1_type=matching_d1_type,
                    doc1_value=f"[{role1} Document]",
                    doc2_type=matching_d2_type,
                    doc2_value=f"[{role2} Document]",
                    comparison_result="NOT_APPLICABLE",
                    severity="INFO",
                    message=f"Role separation: '{matching_d1_type}' ({role1}) and '{matching_d2_type}' ({role2}) belong to different entity roles in Education Loan.",
                    confidence=1.0
                ))
                na_count += 1
                continue

            doc1_obj = docs_by_type[matching_d1_type]
            doc2_obj = docs_by_type[matching_d2_type]

            # CANONICAL ACCESSOR LOGIC
            val1, key1 = get_canonical_value(doc1_obj, fname)
            val2, key2 = get_canonical_value(doc2_obj, fname)

            logger.info(f"Canonical extraction for field='{fname}': doc1({matching_d1_type}.{key1})='{mask_pii(val1, fname)}', doc2({matching_d2_type}.{key2})='{mask_pii(val2, fname)}'")

            # CRITICAL RULE: Missing / null value MUST return UNVERIFIABLE with INFO severity (NEVER MISMATCH)
            if not is_valid_value(val1) or not is_valid_value(val2):
                missing_doc = matching_d1_type if not is_valid_value(val1) else matching_d2_type
                missing_key = key1 if not is_valid_value(val1) else key2
                findings.append(CrossDocumentFinding(
                    field_name=fname,
                    doc1_type=matching_d1_type,
                    doc1_value=val1,
                    doc2_type=matching_d2_type,
                    doc2_value=val2,
                    comparison_result="UNVERIFIABLE",
                    severity="INFO",
                    message=f"Cannot verify '{fname}': field is missing or null in document '{missing_doc}' (key: {missing_key}).",
                    confidence=1.0
                ))
                unverifiable_count += 1
                continue

            # Perform Field Comparison
            res_str = "UNVERIFIABLE"
            conf = 1.0
            msg = ""

            if comp_type == "strict_identifier":
                res_str, conf, msg = compare_strict_identifiers(str(val1), str(val2), fname)
            elif comp_type == "name":
                res_str, conf, msg = compare_names(str(val1), str(val2))
            elif comp_type == "land_owner_vs_farmer":
                res_str, conf, msg = compare_land_owner_vs_farmer(str(val1), str(val2))
            elif comp_type == "address":
                res_str, conf, msg = compare_addresses(str(val1), str(val2))
            elif comp_type == "survey_number":
                res_str, conf, msg = compare_survey_numbers(str(val1), str(val2))
            elif comp_type == "area":
                res_str, conf, msg = compare_areas(val1, val2)
            elif comp_type == "text":
                res_str, conf, msg = compare_text_exact(str(val1), str(val2), fname)
            elif comp_type == "salary":
                num1 = normalize_currency(val1)
                num2 = normalize_currency(val2)
                if num1 is not None and num2 is not None:
                    res_str, conf, msg = compare_salaries(num1, matching_d1_type, num2, matching_d2_type)
                else:
                    res_str = "UNVERIFIABLE"
                    msg = f"Non-numeric salary values cannot be compared: '{val1}' vs '{val2}'"

            # Optional LLM Fallback if MISMATCH or borderline for non-strict fields
            if res_str in ["MISMATCH", "MINOR_VARIATION"] and comp_type not in ["strict_identifier", "survey_number"]:
                llm_res = semantic_llm_compare(fname, val1, matching_d1_type, val2, matching_d2_type)
                if llm_res and llm_res[0] in ["MATCH", "MINOR_VARIATION"]:
                    res_str, conf, msg = llm_res

            # Determine severity
            if res_str == "MATCH":
                sev = "INFO"
                match_count += 1
            elif res_str == "MINOR_VARIATION":
                sev = "LOW"
                minor_var_count += 1
            elif res_str == "MISMATCH":
                sev = sev_mismatch
                mismatch_count += 1
            else:
                sev = "INFO"
                unverifiable_count += 1

            findings.append(CrossDocumentFinding(
                field_name=fname,
                doc1_type=matching_d1_type,
                doc1_value=mask_pii(val1, fname),
                doc2_type=matching_d2_type,
                doc2_value=mask_pii(val2, fname),
                comparison_result=res_str,
                severity=sev,
                message=msg,
                confidence=conf
            ))

        # Calculate Verification Coverage & Consistency Score
        total_comps = len(findings)
        verified_comps = match_count + minor_var_count + mismatch_count
        coverage = round((verified_comps / total_comps * 100.0), 1) if total_comps > 0 else 0.0

        if verified_comps == 0:
            score = None
            status = "INSUFFICIENT_EVIDENCE"
        else:
            # Score calculated strictly from verified comparisons
            score = 100.0
            for f in findings:
                if f.comparison_result == "MISMATCH":
                    if f.severity == "CRITICAL":
                        score -= 25.0
                    elif f.severity == "HIGH":
                        score -= 15.0
                    elif f.severity == "MEDIUM":
                        score -= 8.0
                    elif f.severity == "LOW":
                        score -= 3.0
                elif f.comparison_result == "MINOR_VARIATION":
                    score -= 1.0

            score = max(0.0, min(100.0, round(score, 1)))

            has_critical_or_high_mismatch = any(f.comparison_result == "MISMATCH" and f.severity in ["CRITICAL", "HIGH"] for f in findings)
            has_any_mismatch = any(f.comparison_result == "MISMATCH" for f in findings)

            if has_critical_or_high_mismatch:
                status = "REVIEW_REQUIRED"
            elif coverage < min_coverage:
                status = "PARTIALLY_VERIFIED"
            elif score >= 85.0 and not has_any_mismatch:
                status = "PASS"
            elif score >= 60.0:
                status = "WARNING"
            else:
                status = "FAIL"

        processing_time = round((time.time() - start_time) * 1000, 2)

        result = CrossDocumentResult(
            application_id=application_id,
            loan_type=loan_type,
            consistency_score=score,
            verification_coverage=coverage,
            verification_status=status,
            total_comparisons=total_comps,
            verified_comparisons=verified_comps,
            match_count=match_count,
            minor_variation_count=minor_var_count,
            mismatch_count=mismatch_count,
            unverifiable_count=unverifiable_count,
            not_applicable_count=na_count,
            findings=findings,
            processing_time_ms=processing_time,
            errors=[],
            next_agent="risk_anomaly_agent"
        )

        logger.info(f"Cross-Document Verification Complete: Score={result.consistency_score}, Coverage={result.verification_coverage}%, Status={result.verification_status}, Verified={verified_comps}/{total_comps}")
        return result


# Wrapper Alias Class
InformationCrossDocumentAgent = CrossDocumentVerifier


# =============================================================================
# 8. LANGGRAPH WORKFLOW NODE INTEGRATION
# =============================================================================

def cross_document_verification_node(state: LoanDocumentState) -> LoanDocumentState:
    """
    LangGraph node function executing Agent 4 cross-document verification.
    """
    app_id = state.get("application_id", "APP-LANGGRAPH")
    loan_type = state.get("loan_type", "personal_loan")
    val_results = state.get("validation_results") or state.get("extraction_results") or []

    verifier = CrossDocumentVerifier()
    result = verifier.process(val_results, loan_type=loan_type, application_id=app_id)

    state["cross_document_results"] = result.model_dump()
    state["next_agent"] = "risk_anomaly_agent"
    return state


def build_cross_document_graph() -> StateGraph:
    """
    Builds the LangGraph StateGraph pipeline for Agent 4.
    """
    workflow = StateGraph(LoanDocumentState)
    workflow.add_node("cross_document_agent", cross_document_verification_node)
    workflow.set_entry_point("cross_document_agent")
    workflow.add_edge("cross_document_agent", END)
    return workflow.compile()


if __name__ == "__main__":
    logger.info("Executing Agent 4 self-test...")
    verifier = CrossDocumentVerifier()
    
    sample_docs = [
        {
            "document_type": "kyc_identity",
            "fields": {
                "full_name": {"value": "Ananya Rao"},
                "identity_document_number": {"value": "ABCDE1234F"},
                "address": {"value": "Plot 42, Jubilee Hills, Hyderabad"}
            }
        },
        {
            "document_type": "payslip",
            "fields": {
                "employee_name": {"value": "Ananya Rao"},
                "pan": {"value": "ABCDE1234F"},
                "gross_salary": {"value": 85000.0}
            }
        },
        {
            "document_type": "itr_tax_return",
            "fields": {
                "taxpayer_name": {"value": "Ananya Rao"},
                "pan_number": {"value": "ABCDE1234F"},
                "gross_total_income": {"value": 1020000.0}
            }
        }
    ]
    
    res = verifier.process(sample_docs, loan_type="personal_loan", application_id="TEST-APP-01")
    print(json.dumps(res.model_dump(), indent=2))
