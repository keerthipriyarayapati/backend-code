"""
Central Document Requirement Policy for Multi-Agent Loan Document Processing AI.
Single Source of Truth for all 10 supported loan types across FastAPI, Agents, and React Frontend.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentRequirement(BaseModel):
    """Defines a single document requirement slot for a loan application."""
    requirement_id: str = Field(..., description="Unique ID for this requirement slot")
    display_name: str = Field(..., description="User-facing display name")
    accepted_document_types: List[str] = Field(..., description="Agent 1 classification types accepted by this slot")
    required: bool = Field(default=True, description="Whether this document slot is mandatory")
    description: Optional[str] = Field(default=None, description="Guidance text for user")
    entity_role: Optional[str] = Field(default="PRIMARY", description="Role associated with this requirement (e.g. PRIMARY, STUDENT, CO_APPLICANT, FARMER, LAND_OWNER)")


class DocumentSlotStatus(BaseModel):
    """Represents current upload status of a single requirement slot."""
    requirement_id: str
    display_name: str
    required: bool
    accepted_document_types: List[str]
    status: str = Field(default="pending", description="One of: pending, accepted, wrong_document, duplicate, error")
    uploaded_document_id: Optional[str] = None
    uploaded_filename: Optional[str] = None
    file_path: Optional[str] = None
    detected_document_type: Optional[str] = None
    canonical_document_type: Optional[str] = None
    display_document_type: Optional[str] = None
    is_valid_for_slot: bool = False
    slot_status: str = Field(default="pending", description="One of: pending, accepted, rejected")
    wrong_document: bool = False
    confidence: Optional[float] = None
    error: Optional[str] = None



class ApplicationDocumentStatus(BaseModel):
    """Overall document readiness status for a loan application."""
    application_id: str
    loan_type: str
    application_status: str = Field(default="INCOMPLETE", description="NOT_STARTED, INCOMPLETE, READY_FOR_PROCESSING, PROCESSING, COMPLETED, BLOCKED")
    required_documents_count: int = 0
    uploaded_required_documents_count: int = 0
    missing_required_documents_count: int = 0
    wrong_documents_count: int = 0
    optional_documents_count: int = 0
    uploaded_optional_documents_count: int = 0
    slots: List[DocumentSlotStatus] = Field(default_factory=list)


# =============================================================================
# CENTRAL DOCUMENT REQUIREMENT POLICY (10 SUPPORTED LOAN TYPES)
# =============================================================================

LOAN_TYPE_NAMES: Dict[str, str] = {
    "home_loan": "Home Loan",
    "personal_loan": "Personal Loan",
    "vehicle_loan": "Car / Vehicle Loan",
    "education_loan": "Education Loan",
    "business_loan": "Business Loan",
    "gold_loan": "Gold Loan",
    "loan_against_property": "Loan Against Property",
    "agriculture_loan": "Agriculture / Crop Loan",
    "loan_against_fd": "Loan Against Fixed Deposit",
    "consumer_durable_loan": "Consumer Durable Loan"
}


LOAN_DOCUMENT_POLICY: Dict[str, Dict[str, List[DocumentRequirement]]] = {

    # 1. HOME LOAN
    "home_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="home_loan_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "pan_card", "aadhaar_identity", "passport_identity", "driving_license_identity", "voter_id_identity"],
                required=True,
                description="Government-issued photo identity proof"
            ),
            DocumentRequirement(
                requirement_id="home_loan_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True,
                description="Permanent Account Number (PAN) Card"
            ),
            DocumentRequirement(
                requirement_id="home_loan_income",
                display_name="Income Proof",
                accepted_document_types=["payslip", "salary_certificate", "employment_income_proof", "itr_tax_return", "form_16"],
                required=True,
                description="Payslip, Salary Certificate, or Tax Return"
            ),
            DocumentRequirement(
                requirement_id="home_loan_bank_statement",
                display_name="Bank Statement",
                accepted_document_types=["bank_statement", "business_bank_statement"],
                required=True,
                description="Latest 6 months bank statement"
            ),
            DocumentRequirement(
                requirement_id="home_loan_property_title",
                display_name="Property Title Document",
                accepted_document_types=["property_title_document", "sale_deed", "property_registration_document"],
                required=True,
                description="Registered property ownership deed or title document"
            ),
            DocumentRequirement(
                requirement_id="home_loan_sale_agreement",
                display_name="Sale Agreement",
                accepted_document_types=["sale_agreement"],
                required=True,
                description="Agreement to Sell / Builder Buyer Agreement"
            ),
            DocumentRequirement(
                requirement_id="home_loan_approved_plan",
                display_name="Approved Building Plan",
                accepted_document_types=["approved_building_plan"],
                required=True,
                description="Municipal / Authority approved building construction plan"
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="home_loan_itr",
                display_name="ITR / Tax Return",
                accepted_document_types=["itr_tax_return"],
                required=False,
                description="Income Tax Return computation"
            ),
            DocumentRequirement(
                requirement_id="home_loan_emp_letter",
                display_name="Employment Letter",
                accepted_document_types=["employment_letter", "employment_certificate", "office_id", "employee_id", "company_id", "employment_proof"],
                required=False,
                description="Job offer / confirmation / service letter"
            ),
            DocumentRequirement(
                requirement_id="home_loan_property_tax",
                display_name="Property Tax Receipt",
                accepted_document_types=["property_tax_receipt", "land_tax_receipt"],
                required=False,
                description="Latest paid property tax receipt"
            ),
            DocumentRequirement(
                requirement_id="home_loan_valuation",
                display_name="Property Valuation Report",
                accepted_document_types=["property_valuation_report"],
                required=False,
                description="Chartered engineer / certified valuer report"
            )
        ]
    },

    # 2. PERSONAL LOAN
    "personal_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="personal_loan_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "passport_identity", "driving_license_identity", "voter_id_identity"],
                required=True,
                description="Photo Identity Proof"
            ),
            DocumentRequirement(
                requirement_id="personal_loan_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True,
                description="PAN Card Copy"
            ),
            DocumentRequirement(
                requirement_id="personal_loan_payslip",
                display_name="Payslip / Salary Certificate",
                accepted_document_types=["payslip", "salary_certificate"],
                required=True,
                description="Latest 3 months salary payslip"
            ),
            DocumentRequirement(
                requirement_id="personal_loan_bank_statement",
                display_name="Bank Statement",
                accepted_document_types=["bank_statement"],
                required=True,
                description="Salary account 6 months bank statement"
            ),
            DocumentRequirement(
                requirement_id="personal_loan_emp_proof",
                display_name="Employment Proof",
                accepted_document_types=["employment_letter", "employment_certificate", "office_id", "employee_id", "company_id", "employment_proof"],
                required=True,
                description="Employment letter or office ID"
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="personal_loan_itr",
                display_name="ITR / Tax Return",
                accepted_document_types=["itr_tax_return"],
                required=False
            ),
            DocumentRequirement(
                requirement_id="personal_loan_form16",
                display_name="Form 16",
                accepted_document_types=["form_16"],
                required=False
            )
        ]
    },

    # 3. CAR / VEHICLE LOAN
    "vehicle_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="vehicle_loan_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "passport_identity", "driving_license_identity"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="vehicle_loan_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="vehicle_loan_income",
                display_name="Income Proof",
                accepted_document_types=["payslip", "salary_certificate", "itr_tax_return", "form_16"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="vehicle_loan_bank_statement",
                display_name="Bank Statement",
                accepted_document_types=["bank_statement"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="vehicle_loan_quotation",
                display_name="Vehicle Quotation",
                accepted_document_types=["vehicle_quotation"],
                required=True,
                description="Official dealership price quotation"
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="vehicle_loan_invoice",
                display_name="Vehicle Invoice",
                accepted_document_types=["vehicle_invoice"],
                required=False
            ),
            DocumentRequirement(
                requirement_id="vehicle_loan_insurance",
                display_name="Vehicle Insurance",
                accepted_document_types=["vehicle_insurance_document"],
                required=False
            )
        ]
    },

    # 4. EDUCATION LOAN
    "education_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="edu_loan_student_kyc",
                display_name="Student KYC / Identity",
                accepted_document_types=["student_kyc", "kyc_identity", "aadhaar_identity", "passport_identity"],
                required=True,
                entity_role="STUDENT"
            ),
            DocumentRequirement(
                requirement_id="edu_loan_admission_letter",
                display_name="Admission Letter",
                accepted_document_types=["admission_letter"],
                required=True,
                description="Confirmed admission offer letter from university",
                entity_role="STUDENT"
            ),
            DocumentRequirement(
                requirement_id="edu_loan_fee_structure",
                display_name="Fee Structure",
                accepted_document_types=["fee_structure"],
                required=True,
                description="University fee schedule",
                entity_role="STUDENT"
            ),
            DocumentRequirement(
                requirement_id="edu_loan_academic_proof",
                display_name="Academic Certificate / Marksheet",
                accepted_document_types=["academic_certificate", "marksheet"],
                required=True,
                entity_role="STUDENT"
            ),
            DocumentRequirement(
                requirement_id="edu_loan_co_app_kyc",
                display_name="Co-applicant KYC",
                accepted_document_types=["co_applicant_kyc", "kyc_identity", "pan_card"],
                required=True,
                entity_role="CO_APPLICANT"
            ),
            DocumentRequirement(
                requirement_id="edu_loan_co_app_income",
                display_name="Co-applicant Income Proof",
                accepted_document_types=["co_applicant_income_proof", "payslip", "salary_certificate", "itr_tax_return"],
                required=True,
                entity_role="CO_APPLICANT"
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="edu_loan_co_app_bank",
                display_name="Co-applicant Bank Statement",
                accepted_document_types=["bank_statement"],
                required=False,
                entity_role="CO_APPLICANT"
            )
        ]
    },

    # 5. BUSINESS LOAN
    "business_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="biz_loan_promoter_kyc",
                display_name="Promoter / Applicant KYC",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_registration",
                display_name="Business Registration",
                accepted_document_types=["business_registration", "partnership_deed", "incorporation_certificate", "business_license"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_gst_cert",
                display_name="GST Certificate",
                accepted_document_types=["gst_certificate"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_bank_stmt",
                display_name="Business Bank Statement",
                accepted_document_types=["business_bank_statement", "bank_statement"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_itr_gst_return",
                display_name="ITR / GST Return",
                accepted_document_types=["business_itr", "itr_tax_return", "gst_return"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_pnl",
                display_name="Profit & Loss Statement",
                accepted_document_types=["profit_loss_statement"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="biz_loan_balance_sheet",
                display_name="Balance Sheet",
                accepted_document_types=["balance_sheet"],
                required=True
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="biz_loan_license",
                display_name="Business License",
                accepted_document_types=["business_license"],
                required=False
            )
        ]
    },

    # 6. GOLD LOAN
    "gold_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="gold_loan_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "passport_identity"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="gold_loan_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="gold_loan_security_doc",
                display_name="Gold / Security Document",
                accepted_document_types=["gold_security_document", "gold_loan_document", "security_document"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="gold_loan_valuation",
                display_name="Gold Valuation Report",
                accepted_document_types=["jewellery_valuation_report", "gold_valuation_document"],
                required=True
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="gold_loan_pledge",
                display_name="Pledge Document",
                accepted_document_types=["pledge_document"],
                required=False
            )
        ]
    },

    # 7. LOAN AGAINST PROPERTY
    "loan_against_property": {
        "required": [
            DocumentRequirement(
                requirement_id="lap_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "passport_identity"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="lap_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="lap_property_title",
                display_name="Property Ownership / Title Document",
                accepted_document_types=["property_title_document", "sale_deed", "property_registration_document"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="lap_tax_receipt",
                display_name="Property Tax Receipt",
                accepted_document_types=["property_tax_receipt", "land_tax_receipt"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="lap_valuation",
                display_name="Property Valuation Report",
                accepted_document_types=["property_valuation_report"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="lap_income",
                display_name="Income Proof",
                accepted_document_types=["payslip", "salary_certificate", "itr_tax_return"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="lap_bank",
                display_name="Bank Statement",
                accepted_document_types=["bank_statement"],
                required=True
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="lap_sale_agreement",
                display_name="Sale Agreement",
                accepted_document_types=["sale_agreement"],
                required=False
            )
        ]
    },

    # 8. AGRICULTURE / CROP LOAN
    "agriculture_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="agri_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "voter_id_identity"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="agri_land_record",
                display_name="Land Ownership Record",
                accepted_document_types=["land_record", "land_ownership_document"],
                required=True,
                description="Patta / Chitta / 7/12 land extract"
            ),
            DocumentRequirement(
                requirement_id="agri_cultivation_record",
                display_name="Cultivation / Crop Record",
                accepted_document_types=["cultivation_record", "crop_document"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="agri_income",
                display_name="Agricultural Income Proof",
                accepted_document_types=["agricultural_income_proof"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="agri_bank",
                display_name="Bank Statement",
                accepted_document_types=["bank_statement"],
                required=True
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="agri_tax",
                display_name="Land Tax Receipt",
                accepted_document_types=["land_tax_receipt", "property_tax_receipt"],
                required=False
            )
        ]
    },

    # 9. LOAN AGAINST FIXED DEPOSIT
    "loan_against_fd": {
        "required": [
            DocumentRequirement(
                requirement_id="fd_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=["kyc_identity", "aadhaar_identity", "passport_identity"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="fd_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="fd_certificate",
                display_name="Fixed Deposit Certificate / Receipt",
                accepted_document_types=["fixed_deposit_certificate", "fixed_deposit_receipt"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="fd_bank_doc",
                display_name="Bank Account Document",
                accepted_document_types=["bank_account_document", "bank_statement"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="fd_statement",
                display_name="FD Statement",
                accepted_document_types=["fd_statement", "fixed_deposit_certificate"],
                required=True
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="fd_loan_doc",
                display_name="FD Loan Document",
                accepted_document_types=["fd_loan_document"],
                required=False
            )
        ]
    },

    # 10. CONSUMER DURABLE LOAN
    "consumer_durable_loan": {
        "required": [
            DocumentRequirement(
                requirement_id="cd_kyc",
                display_name="KYC / Identity Proof",
                accepted_document_types=[
                    "kyc_identity",
                    "aadhaar_identity",
                    "passport_identity",
                    "voter_id",
                    "driving_license"
                ],
                required=True,
                description="Government-issued photo identity proof"
            ),
            DocumentRequirement(
                requirement_id="cd_pan",
                display_name="PAN Card",
                accepted_document_types=[
                    "pan_card"
                ],
                required=True,
                description="Permanent Account Number (PAN) Card"
            ),
            DocumentRequirement(
                requirement_id="cd_income",
                display_name="Income Proof",
                accepted_document_types=[
                    "payslip",
                    "salary_certificate",
                    "employment_income_proof",
                    "itr_tax_return"
                ],
                required=True,
                description="Payslip, Salary Certificate, or ITR Tax Return"
            ),
            DocumentRequirement(
                requirement_id="cd_bank",
                display_name="Bank Statement",
                accepted_document_types=[
                    "bank_statement"
                ],
                required=True,
                description="Bank account statement"
            ),
            DocumentRequirement(
                requirement_id="cd_quotation",
                display_name="Product Quotation",
                accepted_document_types=[
                    "product_quotation",
                    "consumer_durable_quotation",
                    "product_invoice"
                ],
                required=True,
                description="Store proforma invoice / quotation"
            )
        ],
        "optional": []
    }
}


CANONICAL_DOCUMENT_TYPES = {
    # KYC / Identity
    "kyc_identity", "student_kyc", "co_applicant_kyc", "aadhaar_identity",
    "pan_card", "passport_identity", "voter_identity", "voter_id", "driving_license",

    # Income / Employment
    "payslip", "salary_certificate", "employment_income_proof", "income_proof",
    "agricultural_income_proof", "itr_tax_return", "form_16",
    "employment_letter", "employment_certificate", "office_id", "employee_id",
    "company_id", "employment_proof", "income_certificate", "co_applicant_income_proof",

    # Banking
    "bank_statement", "business_bank_statement",

    # Property
    "property_title_document", "sale_deed", "sale_agreement",
    "property_registration_document", "property_valuation_report",
    "property_tax_receipt", "approved_building_plan", "land_ownership_document",
    "land_record", "land_tax_receipt", "encumbrance_certificate",

    # Vehicle
    "vehicle_quotation", "vehicle_invoice", "vehicle_purchase_agreement",
    "vehicle_registration_document", "vehicle_insurance_document",

    # Education
    "admission_letter", "fee_structure", "academic_certificate", "marksheet",

    # Business
    "business_registration", "gst_certificate", "gst_return", "business_itr",
    "profit_loss_statement", "balance_sheet", "partnership_deed", "incorporation_certificate",
    "business_license",

    # Gold / FD / Consumer
    "gold_security_document", "jewellery_valuation_report", "pledge_document",
    "fixed_deposit_certificate", "fixed_deposit_receipt", "fd_statement", "fd_loan_document",
    "product_quotation", "consumer_durable_quotation", "product_invoice", "purchase_invoice",

    # Status Types
    "other", "unknown"
}


CANONICAL_LABELS: Dict[str, str] = {
    "student_kyc": "Student KYC / Identity",
    "co_applicant_kyc": "Co-applicant KYC / Identity",
    "kyc_identity": "KYC / Identity Proof",
    "pan_card": "PAN Card",
    "aadhaar_identity": "Aadhaar Card",
    "passport_identity": "Passport",
    "voter_id": "Voter ID",
    "voter_identity": "Voter ID",
    "driving_license": "Driving License",
    "payslip": "Salary Payslip",
    "salary_certificate": "Salary Certificate",
    "employment_income_proof": "Employment Income Proof",
    "income_proof": "Income Proof",
    "itr_tax_return": "Income Tax Return (ITR)",
    "form_16": "Form 16 TDS Certificate",
    "bank_statement": "Bank Statement",
    "business_bank_statement": "Business Bank Statement",
    "product_quotation": "Product Quotation",
    "consumer_durable_quotation": "Consumer Durable Quotation",
    "product_invoice": "Product Invoice",
    "purchase_invoice": "Purchase Invoice",
    "property_title_document": "Property Title Document",
    "sale_deed": "Sale Deed",
    "sale_agreement": "Sale Agreement",
    "property_registration_document": "Property Registration Document",
    "property_valuation_report": "Property Valuation Report",
    "property_tax_receipt": "Property Tax Receipt",
    "approved_building_plan": "Approved Building Plan",
    "land_ownership_document": "Land Ownership Record",
    "agricultural_income_proof": "Agricultural Income Proof",
    "cd_kyc": "KYC / Identity Proof",
    "cd_pan": "PAN Card",
    "cd_income": "Income Proof",
    "cd_bank": "Bank Statement",
    "cd_quotation": "Product Quotation"
}


def get_display_document_type(doc_type: Optional[str]) -> str:
    """Returns clean human-friendly display name for a canonical document type."""
    if not doc_type:
        return "Unknown Document"
    norm = normalize_document_type(doc_type)
    return CANONICAL_LABELS.get(norm, norm.replace("_", " ").title())



def normalize_document_type(raw_type: Optional[str], requirement_id: Optional[str] = None, loan_type: Optional[str] = None) -> str:
    """
    Canonical Document Type Normalization Engine.
    Maps raw classifications, uppercase enum names, and UI labels to standardized canonical identifiers.
    Never silently converts unknown enum values into 'other'.
    """
    if not raw_type:
        return "unknown"

    cleaned = str(raw_type).strip()
    lower = cleaned.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    # Collapse consecutive underscores
    while "__" in lower:
        lower = lower.replace("__", "_")
    lower = lower.strip("_")

    # Consumer Durable Loan canonical normalization
    if lower in ["cd_kyc", "consumer_durable_kyc"]:
        return "kyc_identity"
    if lower in ["cd_pan", "consumer_durable_pan"]:
        return "pan_card"
    if lower in ["cd_income", "consumer_durable_income"]:
        return "income_proof"
    if lower in ["cd_bank", "consumer_durable_bank"]:
        return "bank_statement"
    if lower in ["cd_quotation", "consumer_durable_quotation"]:
        return "product_quotation"

    # Direct canonical match
    if lower in CANONICAL_DOCUMENT_TYPES:
        return lower

    # Specific KYC Normalization
    if lower in ["student_kyc", "student_identity", "student_id", "student_id_card", "student_pan_or_identity_record"]:
        return "student_kyc"
    if lower in ["co_applicant_kyc", "coapplicant_kyc", "co_app_kyc", "co_applicant_identity"]:
        return "co_applicant_kyc"
    if lower in ["pan_card", "pan", "permanent_account_number", "pan_document"]:
        return "pan_card"
    if lower in ["aadhaar_identity", "aadhaar_card", "aadhaar", "uidai", "aadhar", "aadhar_card"]:
        return "aadhaar_identity"
    if lower in ["passport_identity", "passport", "indian_passport"]:
        return "passport_identity"
    if lower in ["voter_id", "voter_identity", "voter_id_identity", "epic_card", "election_card"]:
        return "voter_id"
    if lower in ["driving_license", "driving_licence", "driving_license_identity", "driving_licence_identity", "dl"]:
        return "driving_license"
    if lower in ["kyc_identity", "kyc", "identity_proof", "photo_identity", "government_identity", "sample_identity_proof", "kyc_document"]:
        return "kyc_identity"

    # Specific Property Normalization
    if lower in [
        "property_title_document", "property_title", "title_deed", "property_or_title_documents",
        "property_and_title_documents", "property_documents", "property_document",
        "title_document", "title_documents", "ownership_deed"
    ]:
        return "property_title_document"
    if lower in ["sale_deed", "conveyance_deed", "deed_of_sale"]:
        return "sale_deed"
    if lower in ["property_registration_document", "registration_document", "registered_sale_deed"]:
        return "property_registration_document"
    if lower in ["property_valuation_report", "valuation_report", "property_valuation", "real_estate_valuation"]:
        return "property_valuation_report"
    if lower in ["property_tax_receipt", "property_tax", "municipal_tax_receipt", "tax_receipt"]:
        return "property_tax_receipt"
    if lower in ["land_ownership_document", "land_record", "land_records", "patta", "chitta", "7_12", "7_12_extract"]:
        return "land_ownership_document"
    if lower in ["sale_agreement", "agreement_for_sale", "builder_buyer_agreement"]:
        return "sale_agreement"
    if lower in ["approved_building_plan", "building_plan", "approved_plan", "sanctioned_plan"]:
        return "approved_building_plan"

    # Specific Income Normalization
    if lower in ["agricultural_income_proof", "agricultural_income_document", "agri_income", "agricultural_income"]:
        return "agricultural_income_proof"
    if lower in ["payslip", "salary_slip", "pay_slip", "salary_slip_august", "monthly_payslip"]:
        return "payslip"
    if lower in ["salary_certificate", "salary_letter", "income_certificate", "employment_income_proof"]:
        return "salary_certificate"
    if lower in ["itr_tax_return", "itr", "income_tax_return", "itr_1", "tax_return"]:
        return "itr_tax_return"
    if lower in ["form_16", "form16", "tds_certificate"]:
        return "form_16"
    if lower in ["bank_statement", "account_statement", "bank_details"]:
        return "bank_statement"

    # Requirement Context Resolution for generic KYC
    if lower in ["kyc", "identity"] and requirement_id:
        if "student" in requirement_id:
            return "student_kyc"
        if "co_app" in requirement_id or "coapplicant" in requirement_id:
            return "co_applicant_kyc"

    # Preserve unknown or other explicitly
    if lower in ["other", "others"]:
        return "other"
    if lower in ["unknown", "unreadable", "unsupported", "corrupted"]:
        return "unknown"

    return lower


def get_loan_type_policy(loan_type: str) -> Dict[str, List[DocumentRequirement]]:
    """Returns the document policy requirements (required + optional) for a specified loan type."""
    key = loan_type.lower().strip()
    if key in LOAN_DOCUMENT_POLICY:
        return LOAN_DOCUMENT_POLICY[key]
    # Default fallback to personal loan if loan_type unknown
    return LOAN_DOCUMENT_POLICY["personal_loan"]


def is_document_acceptable_for_requirement(detected_doc_type: str, requirement: DocumentRequirement) -> bool:
    """
    Semantic Resolver: Evaluates whether a detected Agent 1 document classification type
    is acceptable for satisfying a specified document requirement slot.
    Strictly prevents cross-domain pollution (e.g. agricultural income into salary slots).
    """
    if not detected_doc_type or not requirement:
        return False

    det = normalize_document_type(detected_doc_type, requirement.requirement_id)
    accepted_raw = requirement.accepted_document_types or []
    accepted = [normalize_document_type(a) for a in accepted_raw]

    # Exact normalized match
    if det in accepted:
        return True

    # =========================================================================
    # CONSUMER DURABLE LOAN SLOT ENFORCEMENT (EXACT 5 CANONICAL SLOTS)
    # =========================================================================
    if requirement.requirement_id == "cd_kyc":
        return det in {"kyc_identity", "aadhaar_identity", "passport_identity", "voter_id", "voter_identity", "driving_license"}

    if requirement.requirement_id == "cd_pan":
        return det == "pan_card"

    if requirement.requirement_id == "cd_income":
        return det in {"payslip", "salary_certificate", "employment_income_proof", "itr_tax_return", "income_proof"}

    if requirement.requirement_id == "cd_bank":
        return det == "bank_statement"

    if requirement.requirement_id == "cd_quotation":
        return det in {"product_quotation", "consumer_durable_quotation", "product_invoice", "purchase_invoice", "store_invoice"}


    # =========================================================================
    # STRICT AGRICULTURAL INCOME PROOF RULE (CRITICAL DOMAIN GUARD)
    # =========================================================================
    # Agricultural Income Proof is ONLY acceptable for agricultural income slots.
    # It must NEVER be accepted for salary/employment slots (Home Loan, Vehicle Loan, LAP, etc.)
    if det == "agricultural_income_proof":
        return "agricultural_income_proof" in accepted

    # If the slot explicitly requires agricultural income proof, salary proofs are invalid
    if "agricultural_income_proof" in accepted and det not in ["agricultural_income_proof"]:
        return False

    # =========================================================================
    # PROPERTY DOCUMENT BOUNDARY RULES
    # =========================================================================
    # Property Title slots accept property title, sale deed, or property registration
    property_title_family = {"property_title_document", "sale_deed", "property_registration_document"}
    if det in property_title_family and any(a in property_title_family for a in accepted):
        return True

    # Property Valuation Reports and Property Tax Receipts are separate and MUST NOT satisfy Title slots
    if det in ["property_valuation_report", "property_tax_receipt"] and not any(a == det for a in accepted):
        return False

    # Land ownership document (patta/7-12) satisfies land_record slots, NOT residential home loan title slots
    if det in ["land_ownership_document", "land_record"] and any(a in ["land_ownership_document", "land_record"] for a in accepted):
        return True

    # =========================================================================
    # KYC & IDENTITY FAMILY RULES
    # =========================================================================
    kyc_identity_family = {
        "kyc_identity", "aadhaar_identity", "passport_identity",
        "driving_license", "voter_identity"
    }

    # Student KYC slot accepts student_kyc or valid student identity documents
    if requirement.requirement_id == "edu_loan_student_kyc":
        if det in ["student_kyc", "kyc_identity", "aadhaar_identity", "passport_identity", "voter_identity", "driving_license"]:
            return True
        return False

    # Co-applicant KYC slot accepts co_applicant_kyc or co-applicant government ID
    if requirement.requirement_id == "edu_loan_co_app_kyc":
        if det in ["co_applicant_kyc", "kyc_identity", "pan_card", "aadhaar_identity", "passport_identity", "voter_identity", "driving_license"]:
            return True
        return False

    # PAN Card specific slot
    if any(a == "pan_card" for a in accepted):
        if det == "pan_card":
            return True
        return False

    # General KYC / Identity Proof slot
    if any(a in kyc_identity_family or a == "kyc_identity" for a in accepted):
        if det in kyc_identity_family:
            return True
        # PAN card is accepted as KYC identity proof if slot allows
        if det == "pan_card" and any(a == "pan_card" or "kyc" in a for a in accepted):
            return True

    # =========================================================================
    # INCOME / EMPLOYMENT FAMILY RULES
    # =========================================================================
    salary_income_family = {"payslip", "salary_certificate", "employment_income_proof", "itr_tax_return", "form_16"}
    if det in salary_income_family:
        if any(a in salary_income_family or "income" in a for a in accepted):
            return True

    # Employment proof / letter
    employment_proof_family = {"employment_letter", "employment_certificate", "office_id", "employee_id", "company_id", "employment_proof"}
    if det in employment_proof_family and any(a in employment_proof_family for a in accepted):
        return True

    # Bank statement
    if det in ["bank_statement", "business_bank_statement"] and any("bank" in a for a in accepted):
        return True

    return False

