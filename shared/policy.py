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
                accepted_document_types=["kyc_identity", "aadhaar_identity", "voter_id_identity"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="cd_pan",
                display_name="PAN Card",
                accepted_document_types=["pan_card"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="cd_income",
                display_name="Income Proof",
                accepted_document_types=["payslip", "salary_certificate", "itr_tax_return"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="cd_bank",
                display_name="Bank Statement",
                accepted_document_types=["bank_statement"],
                required=True
            ),
            DocumentRequirement(
                requirement_id="cd_quotation",
                display_name="Product Quotation",
                accepted_document_types=["product_quotation"],
                required=True,
                description="Store proforma invoice / quotation"
            )
        ],
        "optional": [
            DocumentRequirement(
                requirement_id="cd_invoice",
                display_name="Product Invoice",
                accepted_document_types=["product_invoice", "purchase_invoice"],
                required=False
            )
        ]
    }
}


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
    """
    if not detected_doc_type or not requirement:
        return False

    det = detected_doc_type.lower().strip()
    accepted = [a.lower().strip() for a in requirement.accepted_document_types]

    if det in accepted:
        return True

    # Generic semantic aliasing rules
    if det == "pan_card" and any(a == "pan_card" or a == "pan" or a.startswith("pan_") for a in accepted):
        return True
    if det in ["payslip", "salary_certificate", "employment_income_proof", "itr_tax_return", "form_16"] and any(
        kw in a for a in accepted for kw in ["income", "payslip", "salary", "itr", "form_16"]
    ):
        return True
    if det in ["kyc_identity", "aadhaar_identity", "passport_identity", "driving_license_identity", "voter_id_identity"] and any(
        kw in a for a in accepted for kw in ["kyc", "identity", "pan", "aadhaar", "passport"]
    ):
        return True
    if det in ["employment_letter", "employment_certificate", "office_id", "employee_id", "company_id", "employment_proof"] and any(
        kw in a for a in accepted for kw in ["employment", "emp", "letter", "certificate", "office_id", "employee_id", "company_id"]
    ):
        return True

    return False
