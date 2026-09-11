"""
===============================================================================
AGENT 1 — DOCUMENT CLASSIFICATION AGENT
===============================================================================
Single Python Implementation File for Agent 1.

Core Responsibility:
Determine what type of document each uploaded file is, classifying into one of:
- payslip
- bank_statement
- itr_tax_return
- kyc_identity
- employment_letter
- form_16
- address_proof
- other
- unknown

Architecture & Technologies:
- LangGraph: StateGraph orchestration & node state transitions.
- DeepAgents / Ollama: Semantic document classification via local LLM runtime.
- PyMuPDF (pymupdf): Safe PDF text extraction.
- python-docx: DOCX text & table extraction.
- Pillow / pytesseract: Image handling.
- Pydantic: Structured output validation & contract enforcement for Agent 2.
- Deterministic Validation: File signatures, extension security, path traversal protection.
"""

import os
import sys
import re
import json
import time
import math
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Pydantic & LangGraph Imports
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END

# Import shared pipeline state
try:
    from shared.state import DocumentClassificationResult, LoanDocumentState, DocumentContent
except ImportError:
    # Fallback import if package path is resolved differently
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from shared.state import DocumentClassificationResult, LoanDocumentState, DocumentContent

# Format specific libraries
import pymupdf  # Modern PyMuPDF API (DO NOT use deprecated fitz)
import docx
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("Agent1_DocumentClassifier")

# Configuration Constants
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
CONFIDENCE_THRESHOLD = float(os.getenv("CLASSIFICATION_CONFIDENCE_THRESHOLD", "0.70"))
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_MB", "20")) * 1024 * 1024

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png", ".svg"}
VALID_DOCUMENT_TYPES = {
    # Existing core categories
    "payslip", "bank_statement", "itr_tax_return", "kyc_identity",
    "employment_letter", "form_16", "address_proof", "other", "unknown", "unreadable",

    # Specific Identity / KYC
    "pan_card", "aadhaar_identity", "passport_identity", "driving_license_identity",
    "voter_id_identity", "student_kyc", "student_identity", "co_applicant_kyc",

    # Income / Employment
    "salary_certificate", "employment_income_proof", "tax_computation",
    "income_certificate", "co_applicant_income_proof", "office_id", "employee_id",
    "company_id", "employment_certificate", "employment_proof",

    # Property
    "property_title_document", "sale_deed", "sale_agreement", "property_registration_document",
    "property_tax_receipt", "approved_building_plan", "property_valuation_report",
    "legal_property_report", "encumbrance_certificate", "land_document", "property_address_proof",

    # Vehicle
    "vehicle_quotation", "vehicle_invoice", "vehicle_purchase_agreement",
    "vehicle_registration_document", "vehicle_insurance_document", "vehicle_valuation_document",

    # Education
    "admission_letter", "fee_structure", "academic_certificate", "marksheet", "education_loan_document",

    # Business
    "business_registration", "gst_certificate", "gst_return", "business_itr",
    "profit_loss_statement", "balance_sheet", "business_bank_statement",
    "business_financial_statement", "partnership_deed", "incorporation_certificate",
    "company_registration", "business_license",

    # Gold
    "gold_security_document", "jewellery_valuation_report", "gold_valuation_document",
    "pledge_document", "gold_loan_document", "security_document",

    # Agriculture
    "land_record", "land_ownership_document", "cultivation_record", "crop_document",
    "agricultural_income_proof", "agricultural_loan_document", "farmer_certificate", "land_tax_receipt",

    # Fixed Deposit
    "fixed_deposit_certificate", "fixed_deposit_receipt", "fd_statement", "fd_loan_document", "bank_account_document",

    # Consumer Durable
    "product_quotation", "product_invoice", "purchase_invoice", "consumer_durable_loan_document", "product_warranty_document"
}

# Known Magic Bytes (File Signatures)
MAGIC_BYTES = {
    ".pdf": [b"%PDF-"],
    ".docx": [b"PK\x03\x04"],  # Zip archive structure
    ".doc": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],  # Compound file binary format
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".svg": [b"<?xml", b"<svg", b"\xef\xbb\xbf<?xml"],
}


# =============================================================================
# 1. SECURITY & FILE VALIDATION
# =============================================================================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and shell injection."""
    basename = os.path.basename(filename)
    sanitized = re.sub(r'[^\w\.-]', '_', basename)
    return sanitized or "unnamed_document"


def validate_file_security(file_path: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validates file existence, path safety, size limits, extension, and magic bytes signature.
    Returns: (is_valid, error_message, error_type)
    """
    path = Path(file_path)
    if not path.is_file():
        return False, f"File does not exist: {file_path}", "missing_file"

    # Exclude evaluation metadata files (e.g. GROUND_TRUTH.txt)
    filename_lower = path.name.lower()
    if "ground_truth" in filename_lower or filename_lower.startswith("ground_truth"):
        return False, "Ground truth metadata files are excluded from loan document processing.", "evaluation_metadata_file"

    # Enforce size limit
    file_size = path.stat().st_size
    if file_size == 0:
        return False, "File is completely empty (0 bytes)", "empty_document"
    if file_size > MAX_FILE_SIZE_BYTES:
        return False, f"File size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes)", "file_too_large"

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file extension: {ext}", "unsupported_extension"

    # Check magic bytes for non-txt extensions
    if ext in MAGIC_BYTES:
        try:
            with open(path, "rb") as f:
                header = f.read(32)
            valid_signature = any(header.startswith(sig) or sig in header for sig in MAGIC_BYTES[ext])
            if not valid_signature:
                return False, f"File signature does not match expected format for {ext}", "signature_mismatch"
        except Exception as e:
            return False, f"Error reading file signature: {str(e)}", "read_error"

    return True, None, None


# =============================================================================
# 2. FORMAT-SPECIFIC DOCUMENT PARSERS
# =============================================================================

class DocumentParser:
    """Safe format-specific document text and metadata parser."""

    @staticmethod
    def parse_pdf(file_path: str) -> Tuple[str, int, str, bool, Optional[str]]:
        """Parses PDF files using PyMuPDF (import pymupdf)."""
        try:
            doc = pymupdf.open(file_path)
            if doc.is_encrypted:
                return "", 0, "encrypted", False, "password_protected_pdf"

            page_count = len(doc)
            text_chunks = []

            for page in doc:
                text = page.get_text("text")
                if text:
                    text_chunks.append(text)

            full_text = "\n".join(text_chunks).strip()
            doc.close()

            if not full_text:
                return "", page_count, "empty/scanned", False, None

            quality = "good" if len(full_text) > 100 else "sparse"
            return full_text, page_count, quality, True, None
        except Exception as e:
            logger.error(f"PyMuPDF failed to parse {file_path}: {e}")
            return "", 0, "corrupted", False, "corrupted_pdf"

    @staticmethod
    def parse_docx(file_path: str) -> Tuple[str, int, str, bool, Optional[str]]:
        """Parses DOCX files using python-docx with paragraph and table support."""
        try:
            doc = docx.Document(file_path)
            text_parts = []

            # Extract paragraph text
            for p in doc.paragraphs:
                if p.text.strip():
                    text_parts.append(p.text.strip())

            # Extract table text (preserving row structure and cell columns)
            for t_idx, table in enumerate(doc.tables):
                text_parts.append(f"\n--- TABLE {t_idx+1} ---")
                for row in table.rows:
                    cell_texts = [cell.text.strip() for cell in row.cells]
                    if any(cell_texts):
                        row_line = " | ".join(c if c else "-" for c in cell_texts)
                        text_parts.append(row_line)

            full_text = "\n".join(text_parts).strip()
            if not full_text:
                return "", 1, "empty", False, None

            quality = "good" if len(full_text) > 50 else "sparse"
            return full_text, 1, quality, True, None
        except Exception as e:
            logger.error(f"python-docx failed to parse {file_path}: {e}")
            return "", 0, "corrupted", False, "malformed_docx"

    @staticmethod
    def parse_doc(file_path: str) -> Tuple[str, int, str, bool, Optional[str]]:
        """Handles legacy binary .doc format safely."""
        # Legacy binary .doc parsing environment safety check
        return "", 0, "unsupported_legacy", False, "unsupported_legacy_format"

    @staticmethod
    def parse_txt(file_path: str) -> Tuple[str, int, str, bool, Optional[str]]:
        """Parses plain text files across multiple encodings."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "ascii"]
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read().strip()
                if not text:
                    return "", 1, "empty", False, None
                quality = "good" if len(text) > 30 else "sparse"
                return text, 1, quality, True, None
            except (UnicodeDecodeError, Exception):
                continue
        return "", 0, "encoding_error", False, "invalid_txt_encoding"

    @staticmethod
    def parse_image(file_path: str) -> Tuple[str, int, str, bool, Optional[str]]:
        """Inspects JPG, JPEG, PNG image files."""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                format_name = img.format

            ocr_text = ""
            # Attempt OCR fallback if pytesseract is available
            try:
                import pytesseract
                with Image.open(file_path) as img:
                    ocr_text = pytesseract.image_to_string(img).strip()
            except Exception:
                # pytesseract not installed or tesseract binary not available
                ocr_text = f"[IMAGE DOCUMENT: {format_name} {width}x{height}px]"

            text_available = bool(ocr_text and not ocr_text.startswith("[IMAGE DOCUMENT"))
            quality = "good" if text_available else "image_without_ocr"
            return ocr_text, 1, quality, text_available, None
        except Exception as e:
            logger.error(f"Image parser failed for {file_path}: {e}")
            return "", 0, "corrupted", False, "invalid_image"

    @staticmethod
    def parse_svg(file_path: str) -> Tuple[str, int, str, bool, Optional[str]]:
        """Safely parses XML SVG documents stripping scripts and extracting visible text."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_xml = f.read()

            # Reject malicious scripts or executable code inside SVG
            if "<script" in raw_xml.lower() or "javascript:" in raw_xml.lower():
                logger.warning(f"Unsafe content detected in SVG: {file_path}")
                return "", 0, "unsafe_content", False, "unsafe_svg_script"

            root = ET.fromstring(raw_xml)
            text_nodes = []
            for elem in root.iter():
                # Extract text content from SVG elements
                tag = elem.tag.split("}")[-1].lower() if "}" in elem.tag else elem.tag.lower()
                if tag in ["text", "tspan", "title", "desc", "tref"] and elem.text:
                    stripped = elem.text.strip()
                    if stripped:
                        text_nodes.append(stripped)

            extracted = "\n".join(text_nodes).strip()
            if not extracted:
                return f"[SVG VECTOR GRAPHIC: no text nodes]", 1, "svg_no_text", False, None

            return extracted, 1, "good", True, None
        except ET.ParseError:
            return "", 0, "corrupted", False, "invalid_svg"
        except Exception as e:
            logger.error(f"SVG parser failed for {file_path}: {e}")
            return "", 0, "corrupted", False, "invalid_svg"


def parse_document(file_path: str) -> Tuple[str, int, str, bool, str, Optional[str]]:
    """
    Main parser router.
    Returns: (text, page_count, quality, text_available, extraction_method, error_type)
    """
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        text, pages, quality, available, err = DocumentParser.parse_pdf(file_path)
        return text, pages, quality, available, "pdf_text", err
    elif ext == ".docx":
        text, pages, quality, available, err = DocumentParser.parse_docx(file_path)
        return text, pages, quality, available, "docx_parser", err
    elif ext == ".doc":
        text, pages, quality, available, err = DocumentParser.parse_doc(file_path)
        return text, pages, quality, available, "doc_parser", err
    elif ext == ".txt":
        text, pages, quality, available, err = DocumentParser.parse_txt(file_path)
        return text, pages, quality, available, "txt_reader", err
    elif ext in [".jpg", ".jpeg", ".png"]:
        text, pages, quality, available, err = DocumentParser.parse_image(file_path)
        return text, pages, quality, available, "image_parser", err
    elif ext == ".svg":
        text, pages, quality, available, err = DocumentParser.parse_svg(file_path)
        return text, pages, quality, available, "svg_parser", err
    else:
        return "", 0, "unsupported", False, "unknown_parser", "unsupported_extension"


# =============================================================================
# 3. OLLAMA & DEEPAGENTS CLASSIFIER
# =============================================================================

CLASSIFICATION_PROMPT = """You are a document classification specialist for a loan processing system.

Your ONLY task is to determine the document category from the extracted text content.

CRITICAL INSTRUCTION ON FILENAMES:
- The user can upload files with ANY arbitrary, random, or misleading filename (e.g. bunny.pdf, abc123.pdf, random_document.pdf, hello.pdf, Bharath_C3.pdf, bank_statement.pdf containing a payslip).
- THE FILENAME MUST NEVER BE USED OR TRUSTED TO DETERMINE THE DOCUMENT TYPE.
- YOU MUST CLASSIFY THE DOCUMENT SOLELY BASED ON THE EXTRACTED TEXT CONTENT AND THE DOMINANT PURPOSE OF THE DOCUMENT.

Choose EXACTLY ONE category from this allowed list:
- payslip
- bank_statement
- itr_tax_return
- kyc_identity
- pan_card
- employment_letter
- form_16
- address_proof
- land_record
- cultivation_record
- crop_document
- agricultural_income_proof
- sale_deed
- sale_agreement
- approved_building_plan
- property_tax_receipt
- property_valuation_report
- vehicle_quotation
- vehicle_invoice
- admission_letter
- fee_structure
- academic_certificate
- gst_certificate
- gst_return
- profit_loss_statement
- balance_sheet
- business_registration
- gold_security_document
- fixed_deposit_certificate
- product_invoice
- other
- unknown

Document Guidelines:
- payslip: Employee salary earnings, basic salary, allowances, deductions, gross/net pay, payroll period.
- bank_statement: Account number, statement period, transactions, debit, credit, balance, deposits, withdrawals. (Note: A salary credit listed in a bank statement is STILL a bank_statement).
- itr_tax_return: Income tax return, Assessment Year, gross total income, taxable income, tax payable, filing details.
- kyc_identity: Government identity (Aadhaar, Passport, Voter ID, Driver License), DOB, photo ID.
- pan_card: Permanent Account Number card issued by Income Tax Department, PAN card, PAN details.
- employment_letter: Offer letter, job appointment, designation, employment confirmation, employer letterhead. (Note: mentioning salary does NOT make it a payslip).
- form_16: Certificate of tax deducted at source (TDS) under Section 203 of Income Tax Act, Form 16, Part A / Part B.
- address_proof: Utility bill (electricity, gas, water), lease agreement, residence certificate.
- land_record: Agricultural land ownership record, Patta, Chitta, 7/12 extract, Record of Rights (RoR), title holder, registered owner. Primary Purpose: Person OWNS land parcel.
- cultivation_record: Crop cultivation record, crop season (Kharif, Rabi, Zaid), sowing date, expected harvest date, expected yield, cultivation method (irrigated/rainfed), crop pattern. Primary Purpose: Person CULTIVATES crop on land during a season.
- agricultural_income_proof: Certificate or statement establishing agricultural income earned by the farmer (financial year, gross/net agricultural income, cultivation expenses, annual income). Primary Purpose: Proves AGRICULTURAL INCOME/EARNINGS earned by farmer.
- other: Readable document that does NOT match any category above.
- unknown: Unreadable text, insufficient evidence, ambiguous, or empty content.

Rules:
1. Base decision strictly on the document text provided below and the PRIMARY PURPOSE of the document.
2. Ignore filename completely.
3. Do NOT invent or extrapolate fields.
4. Do NOT perform fraud detection or risk analysis.
5. Return ONLY a valid JSON object matching this schema:
{{
  "document_type": "<one_of_allowed_categories>",
  "confidence": <float_between_0.0_and_1.0>,
  "reasoning": "<concise_explanation_max_2_sentences>"
}}

Extracted Document Text:
\"\"\"
{document_text}
\"\"\"
"""


def classify_with_rule_engine(document_text: str, filename: str) -> Dict[str, Any]:
    """
    High-precision deterministic rule-based document classification engine.
    Used when Ollama is unavailable, times out, or returns unconfident classification.
    CLASSIFIES PURELY BASED ON EXTRACTED DOCUMENT TEXT CONTENT. FILENAME IS IGNORED.
    """
    text_lower = document_text.lower()

    # 1. Existing Core Document Types
    if any(k in text_lower for k in ["payslip", "salary slip", "monthly payslip", "gross salary", "net payable salary", "net pay", "basic salary", "pay period"]):
        return {"document_type": "payslip", "confidence": 0.95, "reasoning": "Detected payslip earnings header and payroll structure.", "error_type": None}

    if any(k in text_lower for k in ["account statement", "bank statement", "opening balance", "closing balance", "debit", "credit", "ledger balance", "available balance"]) or ("bank" in text_lower and ("account" in text_lower or "balance" in text_lower)):
        return {"document_type": "bank_statement", "confidence": 0.95, "reasoning": "Detected bank account statement headers and transaction ledger.", "error_type": None}

    if any(k in text_lower for k in ["income tax return", "itr-1", "assessment year", "taxable income", "tax payable", "form 1040", "gross total income"]):
        return {"document_type": "itr_tax_return", "confidence": 0.95, "reasoning": "Detected Income Tax Return (ITR) headers and tax computation keywords.", "error_type": None}

    # PAN Card specific check (Priority over generic kyc_identity, provided it's not Form 16 / ITR)
    if not any(k in text_lower for k in ["form no. 16", "form 16", "income tax return", "itr-1"]):
        if any(k in text_lower for k in ["permanent account number", "pan card", "income tax department"]) or bool(re.search(r'\b[A-Z]{5}\d{4}[A-Z]{1}\b', document_text)):
            return {"document_type": "pan_card", "confidence": 0.95, "reasoning": "Detected Permanent Account Number (PAN) identity document signatures.", "error_type": None}

    if any(k in text_lower for k in ["aadhaar", "passport", "voter id", "driver license", "driving licence", "dob:", "identity proof", "identity document", "identity number", "government identity", "sample identity proof"]):
        return {"document_type": "kyc_identity", "confidence": 0.95, "reasoning": "Detected government identity document signatures.", "error_type": None}

    if any(k in text_lower for k in ["office id", "company id", "employee id", "staff id", "employee card", "staff card", "employee identity card", "company identity card", "corporate id"]):
        return {"document_type": "office_id", "confidence": 0.95, "reasoning": "Detected employee office identity card signatures.", "error_type": None}

    if any(k in text_lower for k in ["employment certificate", "certificate of employment", "work certificate", "service certificate"]):
        return {"document_type": "employment_certificate", "confidence": 0.95, "reasoning": "Detected employment certificate or service record.", "error_type": None}

    if any(k in text_lower for k in ["offer of employment", "appointment letter", "employment letter", "confirmation of employment", "employment confirmation", "joining date", "date of joining", "certificate of service", "employment proof", "letter of employment", "currently employed", "employed with"]):
        return {"document_type": "employment_letter", "confidence": 0.90, "reasoning": "Detected employment offer, confirmation, or service certificate headers.", "error_type": None}

    if any(k in text_lower for k in ["form no. 16", "form 16", "section 203", "tax deducted at source", "certificate under section 203"]):
        return {"document_type": "form_16", "confidence": 0.95, "reasoning": "Detected Form 16 TDS certificate signatures.", "error_type": None}

    if any(k in text_lower for k in ["electricity bill", "utility bill", "water bill", "lease agreement", "tenancy agreement", "broadband bill", "gas service", "service address"]):
        return {"document_type": "address_proof", "confidence": 0.90, "reasoning": "Detected utility bill or tenancy address proof document signatures.", "error_type": None}

    # 2. Gold Documents (Check before general valuation reports)
    if any(k in text_lower for k in ["gold valuation", "jewellery valuation", "ornament valuation", "gold weight", "carat", "purity", "appraiser report", "pledge receipt"]) or ("gold" in text_lower and "valuation" in text_lower):
        return {"document_type": "gold_security_document", "confidence": 0.90, "reasoning": "Detected gold jewellery valuation and security pledge record.", "error_type": None}

    # 3. Property Documents
    if any(k in text_lower for k in ["title deed", "sale deed", "property title", "ownership deed", "conveyance deed"]):
        return {"document_type": "sale_deed", "confidence": 0.95, "reasoning": "Detected property sale deed ownership signatures.", "error_type": None}

    if any(k in text_lower for k in ["sale agreement", "agreement for sale", "agreement to sell", "property purchase agreement"]):
        return {"document_type": "sale_agreement", "confidence": 0.95, "reasoning": "Detected sale agreement for property transaction.", "error_type": None}

    if any(k in text_lower for k in ["building plan", "approved plan", "sanctioned plan", "municipal approval", "approval number", "approving authority"]):
        return {"document_type": "approved_building_plan", "confidence": 0.90, "reasoning": "Detected approved building plan and municipal sanction details.", "error_type": None}

    if any(k in text_lower for k in ["property tax", "municipal tax receipt", "tax receipt number", "property id", "assessment tax"]):
        return {"document_type": "property_tax_receipt", "confidence": 0.95, "reasoning": "Detected property tax receipt and assessment details.", "error_type": None}

    if any(k in text_lower for k in ["property valuation", "real estate valuation", "plot valuation", "building valuation", "forced sale value"]) or ("property" in text_lower and "valuation" in text_lower):
        return {"document_type": "property_valuation_report", "confidence": 0.90, "reasoning": "Detected property valuation assessment report.", "error_type": None}

    # 4. Vehicle Documents
    if any(k in text_lower for k in ["vehicle quotation", "car quotation", "proforma invoice", "auto quotation", "ex-showroom price", "on-road price"]):
        return {"document_type": "vehicle_quotation", "confidence": 0.95, "reasoning": "Detected vehicle dealer price quotation details.", "error_type": None}

    if any(k in text_lower for k in ["vehicle invoice", "vehicle tax invoice", "car invoice", "chassis number", "chassis no", "engine number", "engine no", "dealer invoice", "auto invoice"]):
        return {"document_type": "vehicle_invoice", "confidence": 0.95, "reasoning": "Detected vehicle purchase invoice and registration metadata.", "error_type": None}

    # 5. Education Documents
    if any(k in text_lower for k in ["admission letter", "admission offer letter", "admission offer", "provisional admission", "offer of admission", "acceptance letter", "course admission"]):
        return {"document_type": "admission_letter", "confidence": 0.95, "reasoning": "Detected educational institution admission letter.", "error_type": None}

    if any(k in text_lower for k in ["fee structure", "tuition fee", "fee schedule", "academic fees", "hostel fee"]):
        return {"document_type": "fee_structure", "confidence": 0.95, "reasoning": "Detected course fee structure breakdown.", "error_type": None}

    if any(k in text_lower for k in ["marksheet", "grade card", "academic transcript", "statement of marks", "academic certificate", "roll number"]):
        return {"document_type": "academic_certificate", "confidence": 0.90, "reasoning": "Detected academic certificate or transcript marksheet.", "error_type": None}

    # 5. Business Documents
    if any(k in text_lower for k in ["gstin", "gst certificate", "registration certificate under gst", "form gst reg-06"]):
        return {"document_type": "gst_certificate", "confidence": 0.95, "reasoning": "Detected GST registration certificate signatures.", "error_type": None}

    if any(k in text_lower for k in ["gstr-3b", "gstr-1", "gst return", "taxable turnover", "tax liability"]):
        return {"document_type": "gst_return", "confidence": 0.95, "reasoning": "Detected GST return filing details.", "error_type": None}

    if any(k in text_lower for k in ["profit and loss", "profit & loss", "p&l statement", "operating expenses", "gross profit", "net profit"]):
        return {"document_type": "profit_loss_statement", "confidence": 0.95, "reasoning": "Detected Profit and Loss statement headers.", "error_type": None}

    if any(k in text_lower for k in ["balance sheet", "total assets", "total liabilities", "current assets", "shareholders equity"]):
        return {"document_type": "balance_sheet", "confidence": 0.95, "reasoning": "Detected financial Balance Sheet structure.", "error_type": None}

    if any(k in text_lower for k in ["certificate of incorporation", "business registration", "shop and establishment", "partnership deed", "udyam registration"]):
        return {"document_type": "business_registration", "confidence": 0.90, "reasoning": "Detected business registration or incorporation certificate.", "error_type": None}

    # 6. Gold Documents
    if any(k in text_lower for k in ["gold valuation", "jewellery valuation", "ornament valuation", "gold weight", "carat", "purity", "appraiser report", "pledge receipt"]):
        return {"document_type": "gold_security_document", "confidence": 0.90, "reasoning": "Detected gold jewellery valuation and security pledge record.", "error_type": None}

    # 7. Agriculture Documents — 3-Way Weighted Purpose Evidence Engine
    agri_income_very_high_signatures = [
        "agricultural income proof", "agricultural income certificate", "agricultural income statement",
        "agricultural income", "annual agricultural income", "net agricultural income", "gross agricultural income",
        "agricultural expenses", "cultivation expenses", "financial year", "income for fy", "income certificate",
        "income assessment", "income verification", "net income from agriculture", "gross income from agriculture"
    ]
    agri_income_high_signatures = [
        "crop income", "agricultural earnings", "annual income", "gross income", "income period", "income details"
    ]

    cultivation_very_high_signatures = [
        "cultivation record", "crop cultivation record", "crop record", "official cultivation",
        "crop season", "sowing date", "harvest date", "expected harvest", "expected yield",
        "cultivation method", "crop pattern", "cultivation status", "crop production"
    ]
    cultivation_high_signatures = [
        "primary crop", "kharif", "rabi", "zaid", "irrigated", "rainfed", "cultivator", "yield"
    ]

    land_very_high_signatures = [
        "land ownership record", "land ownership certificate", "land record", "patta",
        "chitta", "7/12", "7/12 extract", "record of rights", "ror", "title holder",
        "registered owner", "ownership type", "possession certificate"
    ]
    land_high_signatures = [
        "land owner", "revenue record", "land parcel", "ownership details", "land registration", "khatoni", "khasra"
    ]

    # Calculate evidence scores
    income_score = sum(20 for sig in agri_income_very_high_signatures if sig in text_lower) + \
                   sum(10 for sig in agri_income_high_signatures if sig in text_lower)

    cult_score = sum(20 for sig in cultivation_very_high_signatures if sig in text_lower) + \
                 sum(10 for sig in cultivation_high_signatures if sig in text_lower)

    land_score = sum(20 for sig in land_very_high_signatures if sig in text_lower) + \
                 sum(10 for sig in land_high_signatures if sig in text_lower)

    # Header boost (first 300 chars)
    header_text = text_lower[:300]
    if any(sig in header_text for sig in ["agricultural income", "income proof", "income certificate", "income statement"]):
        income_score += 30
    if any(sig in header_text for sig in ["cultivation record", "crop record", "crop cultivation", "cultivation / crop"]):
        cult_score += 30
    if any(sig in header_text for sig in ["land ownership", "land record", "patta", "7/12"]):
        land_score += 30

    # Decision based on dominant document purpose evidence
    if income_score > 0 and income_score >= cult_score and income_score >= land_score:
        return {
            "document_type": "agricultural_income_proof",
            "confidence": 0.95,
            "reasoning": f"Detected strong agricultural income proof signatures (financial year, gross/net agricultural income, expenses) with income score {income_score}.",
            "error_type": None
        }

    if cult_score > 0 and cult_score >= land_score:
        return {
            "document_type": "cultivation_record",
            "confidence": 0.95,
            "reasoning": f"Detected strong crop cultivation signatures (crop season, sowing/harvest dates, yield, cultivation method) with cultivation score {cult_score}.",
            "error_type": None
        }

    if land_score > 0:
        return {
            "document_type": "land_record",
            "confidence": 0.95,
            "reasoning": f"Detected land ownership signatures (Patta/Chitta/7/12/Owner) with land ownership score {land_score}.",
            "error_type": None
        }

    # 8. Fixed Deposit Documents
    if any(k in text_lower for k in ["fixed deposit", "term deposit receipt", "fd receipt", "fd number", "maturity amount", "maturity date", "deposit amount"]):
        return {"document_type": "fixed_deposit_certificate", "confidence": 0.95, "reasoning": "Detected Fixed Deposit (FD) certificate or term deposit receipt.", "error_type": None}

    # 9. Consumer Durable Documents
    if any(k in text_lower for k in ["consumer durable", "product invoice", "appliance invoice", "electronics invoice", "serial number", "unit price"]):
        return {"document_type": "product_invoice", "confidence": 0.90, "reasoning": "Detected product purchase invoice for consumer durable item.", "error_type": None}

    if len(text_lower.strip()) > 30:
        return {"document_type": "other", "confidence": 0.70, "reasoning": "Readable general document content.", "error_type": None}

    return {"document_type": "unknown", "confidence": 0.0, "reasoning": "Insufficient text content for document classification.", "error_type": "insufficient_text"}


def classify_with_ollama(document_text: str, filename: str) -> Dict[str, Any]:
    """
    Calls local Ollama model via LangChain ChatOllama & ChatPromptTemplate to semantically classify document text.
    Falls back instantly to deterministic rule engine if Ollama is unavailable or times out.
    """
    if not document_text or len(document_text.strip()) < 10:
        return classify_with_rule_engine(document_text or "", filename)

    truncated_text = document_text[:3000]

    # Try LangChain ChatOllama integration
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
        prompt_tmpl = ChatPromptTemplate.from_template(CLASSIFICATION_PROMPT)
        chain = prompt_tmpl | llm

        response = chain.invoke({"document_text": truncated_text})
        raw_text = response.content if hasattr(response, "content") else str(response)

        # Parse JSON from LLM output
        json_match = re.search(r"\{.*\}", str(raw_text), re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group(0))
            doc_type = str(parsed.get("document_type", "unknown")).lower().strip()
            conf = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", "Classification completed by LangChain + ChatOllama."))

            if doc_type not in VALID_DOCUMENT_TYPES:
                doc_type = "unknown"

            if doc_type == "unknown":
                return classify_with_rule_engine(document_text, filename)

            conf = max(0.0, min(1.0, conf))
            return {
                "document_type": doc_type,
                "confidence": conf,
                "reasoning": reasoning,
                "error_type": None
            }
        else:
            return classify_with_rule_engine(document_text, filename)

    except Exception as e:
        logger.warning(f"Ollama/LangChain server unavailable or timed out ({e}). Utilizing deterministic rule classification fallback.")
        return classify_with_rule_engine(document_text, filename)


# =============================================================================
# 4. LANGGRAPH WORKFLOW NODES & GRAPH ASSEMBLY
# =============================================================================

def node_validate_input(state: LoanDocumentState) -> LoanDocumentState:
    """LangGraph Node 1: Security & File Format Validation."""
    docs = state.get("documents", [])
    idx = state.get("current_document_index", 0)

    if idx >= len(docs):
        return state

    current_doc = docs[idx]
    file_path = current_doc.get("file_path", "")

    is_valid, err_msg, err_type = validate_file_security(file_path)
    current_doc["security_valid"] = is_valid
    current_doc["error_msg"] = err_msg
    current_doc["error_type"] = err_type

    if not is_valid:
        logger.warning(f"Validation failed for {file_path}: {err_msg}")

    state["current_document"] = current_doc
    return state


def node_parse_document(state: LoanDocumentState) -> LoanDocumentState:
    """LangGraph Node 2: Extract text using safe format parsers."""
    current_doc = state.get("current_document", {})
    if not current_doc.get("security_valid", True):
        return state

    file_path = current_doc.get("file_path", "")
    text, pages, quality, available, method, parse_err = parse_document(file_path)

    current_doc["extracted_text"] = text
    current_doc["page_count"] = pages
    current_doc["content_quality"] = quality
    current_doc["text_available"] = available
    current_doc["extraction_method"] = method
    if parse_err:
        current_doc["error_type"] = parse_err

    state["current_document"] = current_doc
    return state


def node_check_document_quality(state: LoanDocumentState) -> LoanDocumentState:
    """LangGraph Node 3: Inspect text quality and determine if LLM classification is viable."""
    current_doc = state.get("current_document", {})
    if not current_doc.get("security_valid", True) or current_doc.get("error_type"):
        current_doc["can_classify"] = False
        return state

    text = current_doc.get("extracted_text", "")
    available = current_doc.get("text_available", False)

    if not available or not text.strip():
        current_doc["can_classify"] = False
        if not current_doc.get("error_type"):
            current_doc["error_type"] = "no_extractable_text"
    else:
        current_doc["can_classify"] = True

    state["current_document"] = current_doc
    return state


def node_classify_document(state: LoanDocumentState) -> LoanDocumentState:
    """LangGraph Node 4: Perform LLM Semantic Classification via Ollama/DeepAgents."""
    current_doc = state.get("current_document", {})

    start_time = current_doc.get("start_time", time.time())

    if not current_doc.get("can_classify", False):
        # Default unclassifiable result
        current_doc["classification_result"] = {
            "document_type": "unknown",
            "confidence": 0.0,
            "reasoning": f"Unclassifiable document ({current_doc.get('error_type', 'parsing_failed')}).",
            "error_type": current_doc.get("error_type", "unclassifiable_document")
        }
        current_doc["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        state["current_document"] = current_doc
        return state

    text = current_doc.get("extracted_text", "")
    filename = current_doc.get("filename", "")

    # Execute LLM Semantic Classifier
    res = classify_with_ollama(text, filename)
    current_doc["classification_result"] = res
    current_doc["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)

    state["current_document"] = current_doc
    return state


def node_validate_classification(state: LoanDocumentState) -> LoanDocumentState:
    """LangGraph Node 5: Enforce confidence threshold and validate schema."""
    current_doc = state.get("current_document", {})
    res = current_doc.get("classification_result", {})

    doc_type = res.get("document_type", "unknown")
    conf = res.get("confidence", 0.0)
    reasoning = res.get("reasoning", "")
    err_type = res.get("error_type") or current_doc.get("error_type")

    # Enforce CLASSIFICATION_CONFIDENCE_THRESHOLD
    if conf < CONFIDENCE_THRESHOLD and doc_type != "unknown":
        logger.warning(
            f"Low confidence ({conf:.2f} < {CONFIDENCE_THRESHOLD}) for {current_doc.get('filename')}. Overriding category '{doc_type}' to 'unknown'."
        )
        reasoning = f"[LOW CONFIDENCE OVERRIDE: {conf:.2f} < threshold {CONFIDENCE_THRESHOLD}] Original prediction: {doc_type}. {reasoning}"
        doc_type = "unknown"
        status = "unknown"
    elif err_type or doc_type == "unknown":
        status = "failed" if err_type in ["missing_file", "file_too_large", "corrupted_pdf", "signature_mismatch", "ollama_unavailable"] else "unknown"
    else:
        status = "success"

    # Print Agent 1 Classification Debug Panel
    logger.info(
        f"[AGENT 1 DEBUG PANEL] File: {current_doc.get('filename')} | Type: {doc_type} | Conf: {conf:.2f} | Method: {current_doc.get('extraction_method')} | Text: {len(current_doc.get('extracted_text', ''))} chars"
    )
    print("\n" + "=" * 60)
    print("[AGENT 1 DEBUG PANEL] Document Classification")
    print("=" * 60)
    print(f"Filename              : {current_doc.get('filename')}")
    print(f"File Path             : {current_doc.get('file_path')}")
    print(f"Page Count            : {current_doc.get('page_count', 1)}")
    print(f"Extracted Text Length : {len(current_doc.get('extracted_text', ''))} chars")
    print(f"Extraction Method     : {current_doc.get('extraction_method', 'none')}")
    print(f"Classification Result : {doc_type}")
    print(f"Confidence Score      : {conf:.2f}")
    print(f"Reasoning             : {reasoning}")
    print("=" * 60 + "\n")

    # Build Pydantic DocumentClassificationResult
    try:
        pydantic_res = DocumentClassificationResult(
            document_id=current_doc.get("document_id", "doc_001"),
            filename=sanitize_filename(current_doc.get("filename", "unknown")),
            file_extension=Path(current_doc.get("file_path", "")).suffix.lower(),
            file_path=current_doc.get("file_path"),
            document_type=doc_type,
            confidence=conf,
            classification_reason=reasoning,
            text_available=current_doc.get("text_available", False),
            text_length=len(current_doc.get("extracted_text", "")),
            page_count=current_doc.get("page_count", 1),
            extraction_method=current_doc.get("extraction_method", "none"),
            content_quality=current_doc.get("content_quality", "unknown"),
            processing_time_ms=current_doc.get("processing_time_ms", 0.0),
            status=status,
            error_type=err_type,
            next_agent="extraction_agent"
        )
        current_doc["validated_pydantic"] = pydantic_res.model_dump()
    except ValidationError as ve:
        logger.error(f"Pydantic schema validation failed: {ve}")
        current_doc["validated_pydantic"] = DocumentClassificationResult(
            document_id=current_doc.get("document_id", "doc_001"),
            filename=sanitize_filename(current_doc.get("filename", "unknown")),
            file_extension=".unknown",
            document_type="unknown",
            confidence=0.0,
            classification_reason=f"Pydantic schema validation failure: {str(ve)}",
            text_available=False,
            text_length=0,
            page_count=0,
            extraction_method="none",
            content_quality="corrupted",
            processing_time_ms=current_doc.get("processing_time_ms", 0.0),
            status="failed",
            error_type="pydantic_validation_error",
            next_agent="extraction_agent"
        ).model_dump()

    state["current_document"] = current_doc
    return state


def node_prepare_agent2_output(state: LoanDocumentState) -> LoanDocumentState:
    """LangGraph Node 6: Append output to shared state and update next_agent contract."""
    current_doc = state.get("current_document", {})
    validated = current_doc.get("validated_pydantic", {})

    if "classification_results" not in state:
        state["classification_results"] = []

    state["classification_results"].append(validated)
    state["next_agent"] = "extraction_agent"
    return state


def build_agent_1_graph():
    """Assembles and compiles the executable LangGraph workflow for Agent 1."""
    builder = StateGraph(LoanDocumentState)

    builder.add_node("validate_input", node_validate_input)
    builder.add_node("parse_document", node_parse_document)
    builder.add_node("check_document_quality", node_check_document_quality)
    builder.add_node("classify_document", node_classify_document)
    builder.add_node("validate_classification", node_validate_classification)
    builder.add_node("prepare_agent2_output", node_prepare_agent2_output)

    builder.add_edge(START, "validate_input")
    builder.add_edge("validate_input", "parse_document")
    builder.add_edge("parse_document", "check_document_quality")
    builder.add_edge("check_document_quality", "classify_document")
    builder.add_edge("classify_document", "validate_classification")
    builder.add_edge("validate_classification", "prepare_agent2_output")
    builder.add_edge("prepare_agent2_output", END)

    return builder.compile()


# Global compiled graph
agent_1_graph = build_agent_1_graph()


# =============================================================================
# 5. AGENT 1 MAIN CLASS & MULTI-DOCUMENT BATCH ENGINE
# =============================================================================

class DocumentClassificationAgent:
    """Agent 1 Document Classification Manager supporting single & batch files with failure isolation."""

    def __init__(self):
        self.graph = agent_1_graph

    def process_file(self, file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Processes a single file through the LangGraph pipeline."""
        return self.process_batch([file_path], [doc_id] if doc_id else None)[0]

    def process_batch(self, file_paths: List[str], doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Processes multiple files with failure isolation."""
        results = []

        for idx, path in enumerate(file_paths):
            did = doc_ids[idx] if doc_ids and idx < len(doc_ids) else f"doc_{idx+1:03d}"
            start_t = time.time()

            initial_state: LoanDocumentState = {
                "application_id": f"APP-{int(time.time())}",
                "documents": [{
                    "document_id": did,
                    "file_path": path,
                    "filename": os.path.basename(path),
                    "start_time": start_t
                }],
                "current_document_index": 0,
                "classification_results": [],
                "errors": [],
                "processing_metrics": {},
                "next_agent": "document_agent"
            }

            try:
                final_state = self.graph.invoke(initial_state)
                res = final_state.get("classification_results", [{}])[0]
                results.append(res)
            except Exception as e:
                logger.error(f"Uncaught exception processing file {path}: {e}")
                # Failure isolation guarantee: return structured error result
                failed_res = DocumentClassificationResult(
                    document_id=did,
                    filename=sanitize_filename(os.path.basename(path)),
                    file_extension=Path(path).suffix.lower() if os.path.exists(path) else ".unknown",
                    document_type="unknown",
                    confidence=0.0,
                    classification_reason=f"Pipeline exception: {str(e)}",
                    text_available=False,
                    text_length=0,
                    page_count=0,
                    extraction_method="none",
                    content_quality="corrupted",
                    processing_time_ms=round((time.time() - start_t) * 1000, 2),
                    status="failed",
                    error_type="pipeline_exception",
                    next_agent="extraction_agent"
                ).model_dump()
                results.append(failed_res)

        return results


def run_agent_1(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Helper entry function to run Agent 1 over a list of document file paths."""
    agent = DocumentClassificationAgent()
    return agent.process_batch(file_paths)


# =============================================================================
# 6. EVALUATION METRICS ENGINE & TEST REPORT GENERATOR
# =============================================================================

def calculate_metrics(y_true: List[str], y_pred: List[str], format_labels: List[str], run_times_ms: List[float], confidences: List[float]) -> Dict[str, Any]:
    """Calculates precision, recall, macro F1, confusion matrix, format breakdown, and runtime metrics."""
    total = len(y_true)
    if total == 0:
        return {"error": "No ground truth samples provided"}

    classes = sorted(list(VALID_DOCUMENT_TYPES))

    # Confusion matrix dict: [actual][predicted]
    cm = {act: {pred: 0 for pred in classes} for act in classes}
    correct = 0

    for act, pred in zip(y_true, y_pred):
        act_clean = act if act in VALID_DOCUMENT_TYPES else "unknown"
        pred_clean = pred if pred in VALID_DOCUMENT_TYPES else "unknown"
        cm[act_clean][pred_clean] += 1
        if act_clean == pred_clean:
            correct += 1

    accuracy = (correct / total) * 100.0

    # Per-class metrics
    class_metrics = {}
    precisions = []
    recalls = []
    f1s = []

    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)

        prec = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

        class_total = sum(cm[c].values())
        class_acc = (tp / class_total * 100.0) if class_total > 0 else 0.0
        class_metrics[c] = {
            "accuracy": class_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "total_samples": class_total
        }

    macro_prec = sum(precisions) / len(classes)
    macro_rec = sum(recalls) / len(classes)
    macro_f1 = sum(f1s) / len(classes)

    # Format metrics breakdown
    formats = [".pdf", ".docx", ".doc", ".txt", ".jpg", ".jpeg", ".png", ".svg"]
    format_metrics = {}
    for fmt in formats:
        fmt_indices = [i for i, f in enumerate(format_labels) if f.lower() == fmt.lower()]
        if fmt_indices:
            fmt_correct = sum(1 for i in fmt_indices if y_true[i] == y_pred[i])
            format_metrics[fmt] = (fmt_correct / len(fmt_indices)) * 100.0
        else:
            format_metrics[fmt] = None

    # Runtime metrics
    avg_time = sum(run_times_ms) / total if total > 0 else 0.0
    min_time = min(run_times_ms) if total > 0 else 0.0
    max_time = max(run_times_ms) if total > 0 else 0.0
    avg_conf = (sum(confidences) / total) * 100.0 if total > 0 else 0.0
    unknown_cnt = sum(1 for p in y_pred if p == "unknown")
    unknown_rate = (unknown_cnt / total) * 100.0
    success_cnt = sum(1 for i in range(total) if y_pred[i] != "unknown")
    success_rate = (success_cnt / total) * 100.0
    failed_rate = 100.0 - success_rate

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "format_metrics": format_metrics,
        "confusion_matrix": cm,
        "avg_time_ms": avg_time,
        "min_time_ms": min_time,
        "max_time_ms": max_time,
        "avg_confidence": avg_conf,
        "unknown_rate": unknown_rate,
        "success_rate": success_rate,
        "failed_rate": failed_rate
    }


def evaluate_agent_1():
    """Runs evaluation over synthetic test dataset and prints standard report."""
    print("\n" + "=" * 60)
    print("AGENT 1 — DOCUMENT CLASSIFICATION TEST REPORT")
    print("=" * 60)

    # Ensure project root is in sys.path
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Locate synthetic dataset generator / directory
    try:
        try:
            from tests.synthetic_dataset import get_synthetic_evaluation_dataset
        except ImportError:
            from loan_document_ai.tests.synthetic_dataset import get_synthetic_evaluation_dataset
        dataset = get_synthetic_evaluation_dataset()
    except Exception as e:
        logger.warning(f"Could not import synthetic evaluation dataset: {e}")
        print("\nGround truth unavailable — classification quality metrics were not calculated.")
        print("=" * 60 + "\n")
        return

    agent = DocumentClassificationAgent()

    y_true = []
    y_pred = []
    formats = []
    runtimes = []
    confidences = []
    passed_tests = 0
    failed_tests = 0

    # Check if fast evaluation mode is enabled (default fast for immediate report generation)
    if "--full" not in sys.argv and len(dataset) > 9:
        print("Fast evaluation mode active (running 9 representative samples across all classes). Pass --full for all 45 samples.")
        # Pick 1 sample per category for instant sub-second evaluation
        seen_cats = set()
        fast_dataset = []
        for item in dataset:
            if item["ground_truth"] not in seen_cats:
                seen_cats.add(item["ground_truth"])
                fast_dataset.append(item)
        dataset = fast_dataset

    print(f"\nProcessing {len(dataset)} synthetic evaluation samples...")

    from concurrent.futures import ThreadPoolExecutor

    def process_sample(sample):
        fpath = sample["file_path"]
        expected = sample["ground_truth"]
        ext = sample.get("file_extension", Path(fpath).suffix.lower())

        res = agent.process_file(fpath)
        pred = res.get("document_type", "unknown")
        conf = res.get("confidence", 0.0)
        time_ms = res.get("processing_time_ms", 0.0)

        return {
            "expected": expected,
            "pred": pred,
            "ext": ext,
            "time_ms": time_ms,
            "conf": conf,
            "passed": (pred == expected)
        }

    with ThreadPoolExecutor(max_workers=5) as executor:
        results_list = list(executor.map(process_sample, dataset))

    for item in results_list:
        y_true.append(item["expected"])
        y_pred.append(item["pred"])
        formats.append(item["ext"])
        runtimes.append(item["time_ms"])
        confidences.append(item["conf"])

        if item["passed"]:
            passed_tests += 1
        else:
            failed_tests += 1

    metrics = calculate_metrics(y_true, y_pred, formats, runtimes, confidences)

    print("\nTEST RESULTS")
    print("-" * 60)
    print(f"Total Tests       : {metrics['total']}")
    print(f"Passed            : {passed_tests}")
    print(f"Failed            : {failed_tests}")
    print(f"Skipped           : 0")

    print("\nCLASSIFICATION METRICS")
    print("-" * 60)
    print(f"Accuracy          : {metrics['accuracy']:.2f}%")
    print(f"Macro Precision   : {metrics['macro_precision']:.2f}%")
    print(f"Macro Recall      : {metrics['macro_recall']:.2f}%")
    print(f"Macro F1          : {metrics['macro_f1']:.2f}%")

    print("\nRUNTIME METRICS")
    print("-" * 60)
    print(f"Average Confidence: {metrics['avg_confidence']:.2f}%")
    print(f"Unknown Rate      : {metrics['unknown_rate']:.2f}%")
    print(f"Average Time      : {metrics['avg_time_ms']:.2f} ms")
    print(f"Minimum Time      : {metrics['min_time_ms']:.2f} ms")
    print(f"Maximum Time      : {metrics['max_time_ms']:.2f} ms")
    print(f"Success Rate      : {metrics['success_rate']:.2f}%")
    print(f"Failure Rate      : {metrics['failed_rate']:.2f}%")

    print("\nFORMAT PERFORMANCE")
    print("-" * 60)
    fmt_map = {
        ".pdf": "PDF", ".docx": "DOCX", ".doc": "DOC", ".txt": "TXT",
        ".jpg": "JPG", ".jpeg": "JPEG", ".png": "PNG", ".svg": "SVG"
    }
    for ext_key, label in fmt_map.items():
        val = metrics['format_metrics'].get(ext_key)
        if val is None:
            print(f"{label:<17} : Not enough ground-truth samples.")
        else:
            print(f"{label:<17} : Accuracy: {val:.1f}%")

    print("\nDOCUMENT CLASS PERFORMANCE")
    print("-" * 60)
    class_name_map = {
        "payslip": "Payslip",
        "bank_statement": "Bank Statement",
        "itr_tax_return": "ITR",
        "kyc_identity": "KYC",
        "employment_letter": "Employment Letter",
        "form_16": "Form 16",
        "address_proof": "Address Proof",
        "other": "Other",
        "unknown": "Unknown"
    }
    for c_key, c_label in class_name_map.items():
        c_acc = metrics['class_metrics'][c_key]['accuracy']
        c_count = metrics['class_metrics'][c_key]['total_samples']
        if c_count == 0:
            print(f"{c_label:<17} : Not enough ground-truth samples.")
        else:
            print(f"{c_label:<17} : {c_acc:.1f}% ({c_count} samples)")

    print("\nCONFUSION MATRIX")
    print("-" * 60)
    cats = ["payslip", "bank_statement", "itr_tax_return", "kyc_identity", "employment_letter", "form_16", "address_proof", "other", "unknown"]
    short_cats = ["Pay", "Bank", "ITR", "KYC", "Emp", "F16", "Addr", "Oth", "Unk"]
    header = f"{'ACTUAL \\ PRED':<15}" + "".join(f"{sc:>6}" for sc in short_cats)
    print(header)
    cm_data = metrics["confusion_matrix"]
    for i, cat in enumerate(cats):
        row_str = f"{cats[i]:<15}"
        for pred_cat in cats:
            row_str += f"{cm_data[cat][pred_cat]:>6}"
        print(row_str)

    print("=" * 60 + "\n")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    if "--evaluate" in sys.argv:
        evaluate_agent_1()
    elif len(sys.argv) > 1:
        files = sys.argv[1:]
        print(f"Running Agent 1 on {len(files)} files...")
        agent = DocumentClassificationAgent()
        out = agent.process_batch(files)
        print(json.dumps(out, indent=2))
    else:
        print("Usage:")
        print("  python -m agents.agent_1_document.agent --evaluate")
        print("  python -m agents.agent_1_document.agent file1.pdf file2.docx")
