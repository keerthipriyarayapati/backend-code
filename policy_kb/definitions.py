"""
Loan Policy Knowledge Base Definitions.
Provides complete, deterministic credit underwriting policy rules, document requirements,
and required field definitions across all 10 supported loan types.
All policies and rules cite DEMO_POLICY source documentation with sections and pages.
"""

from typing import Dict, Any, List, Optional

ALL_LOAN_POLICIES: List[Dict[str, Any]] = [
    # 1. HOME LOAN
    {
        "policy_id": "POLICY_HOME_LOAN_V1",
        "loan_type": "Home Loan",
        "loan_key": "home_loan",
        "policy_name": "Retail Housing & Mortgage Credit Policy",
        "version": "1.0",
        "description": "Standard underwriting policy for residential home acquisition, construction, and plot loans.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Retail Mortgage Credit Manual 2026",
        "source_section": "Section 4.1 - Residential Housing Eligibility",
        "source_page": 12,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "HL_R01_MIN_INCOME",
                "rule_code": "HL_MIN_INCOME",
                "category": "INCOME",
                "field_name": "monthly_income",
                "operator": "GTE",
                "expected_value": "25000",
                "threshold_value": 25000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum monthly net income for Home Loan must be at least ₹25,000.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Retail Mortgage Credit Manual 2026",
                "source_section": "Section 4.1.1 - Income Thresholds",
                "source_page": 14
            },
            {
                "rule_id": "HL_R02_MIN_AGE",
                "rule_code": "HL_MIN_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "GTE",
                "expected_value": "21",
                "threshold_value": 21.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Primary applicant must be at least 21 years old at loan origination.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Retail Mortgage Credit Manual 2026",
                "source_section": "Section 4.1.2 - Age Parameters",
                "source_page": 15
            },
            {
                "rule_id": "HL_R03_MAX_AGE",
                "rule_code": "HL_MAX_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "LTE",
                "expected_value": "65",
                "threshold_value": 65.0,
                "severity": "HIGH",
                "mandatory": True,
                "error_message": "Primary applicant age must not exceed 65 years at loan maturity.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Retail Mortgage Credit Manual 2026",
                "source_section": "Section 4.1.2 - Age Parameters",
                "source_page": 15
            },
            {
                "rule_id": "HL_R04_MIN_CIBIL",
                "rule_code": "HL_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "cibil_score",
                "operator": "GTE",
                "expected_value": "650",
                "threshold_value": 650.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum credit bureau score of 650 is required for home loan approval.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Retail Mortgage Credit Manual 2026",
                "source_section": "Section 4.1.4 - Bureau Score Guidelines",
                "source_page": 18
            },
            {
                "rule_id": "HL_R05_MAX_FOIR",
                "rule_code": "HL_MAX_FOIR",
                "category": "INCOME",
                "field_name": "foir_ratio",
                "operator": "LTE",
                "expected_value": "60",
                "threshold_value": 60.0,
                "severity": "HIGH",
                "mandatory": False,
                "error_message": "Fixed Obligation to Income Ratio (FOIR) must not exceed 60%.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Retail Mortgage Credit Manual 2026",
                "source_section": "Section 4.2.1 - Debt Service Ratios",
                "source_page": 20
            }
        ],
        "required_documents": [
            {"slot_id": "home_loan_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "home_loan_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True},
            {"slot_id": "home_loan_income", "document_type": "payslip", "display_name": "Income Proof", "required": True},
            {"slot_id": "home_loan_bank_statement", "document_type": "bank_statement", "display_name": "Bank Statement", "required": True},
            {"slot_id": "home_loan_property", "document_type": "property_title_document", "display_name": "Property Documents", "required": True}
        ],
        "required_fields": [
            {"document_type": "pan_card", "field_name": "pan_number", "required": True},
            {"document_type": "pan_card", "field_name": "name", "required": True},
            {"document_type": "payslip", "field_name": "net_salary", "required": True},
            {"document_type": "bank_statement", "field_name": "account_number", "required": True},
            {"document_type": "property_title_document", "field_name": "property_address", "required": True}
        ]
    },

    # 2. PERSONAL LOAN
    {
        "policy_id": "POLICY_PERSONAL_LOAN_V1",
        "loan_type": "Personal Loan",
        "loan_key": "personal_loan",
        "policy_name": "Unsecured Personal Credit Policy",
        "version": "1.0",
        "description": "Underwriting criteria for clean unsecured consumer term loans.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Unsecured Personal Credit Policy 2026",
        "source_section": "Section 3.2 - Clean Consumer Facilities",
        "source_page": 8,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "PL_R01_MIN_INCOME",
                "rule_code": "PL_MIN_INCOME",
                "category": "INCOME",
                "field_name": "monthly_income",
                "operator": "GTE",
                "expected_value": "20000",
                "threshold_value": 20000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum monthly net income of ₹20,000 required for unsecured personal loans.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Unsecured Personal Credit Policy 2026",
                "source_section": "Section 3.2.1 - Net Salary Threshold",
                "source_page": 9
            },
            {
                "rule_id": "PL_R02_MIN_AGE",
                "rule_code": "PL_MIN_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "GTE",
                "expected_value": "21",
                "threshold_value": 21.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Applicant must be at least 21 years of age.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Unsecured Personal Credit Policy 2026",
                "source_section": "Section 3.2.2 - Age Eligibility",
                "source_page": 10
            },
            {
                "rule_id": "PL_R03_MAX_AGE",
                "rule_code": "PL_MAX_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "LTE",
                "expected_value": "60",
                "threshold_value": 60.0,
                "severity": "HIGH",
                "mandatory": True,
                "error_message": "Applicant age must not exceed 60 years at loan maturity.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Unsecured Personal Credit Policy 2026",
                "source_section": "Section 3.2.2 - Age Eligibility",
                "source_page": 10
            },
            {
                "rule_id": "PL_R04_MIN_CIBIL",
                "rule_code": "PL_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "cibil_score",
                "operator": "GTE",
                "expected_value": "700",
                "threshold_value": 700.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Unsecured personal loans require a minimum CIBIL score of 700.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Unsecured Personal Credit Policy 2026",
                "source_section": "Section 3.2.4 - Score Cutoffs",
                "source_page": 12
            }
        ],
        "required_documents": [
            {"slot_id": "pl_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "pl_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True},
            {"slot_id": "pl_income", "document_type": "payslip", "display_name": "Income Proof", "required": True},
            {"slot_id": "pl_bank", "document_type": "bank_statement", "display_name": "Bank Statement", "required": True}
        ],
        "required_fields": [
            {"document_type": "pan_card", "field_name": "pan_number", "required": True},
            {"document_type": "payslip", "field_name": "net_salary", "required": True},
            {"document_type": "bank_statement", "field_name": "account_number", "required": True}
        ]
    },

    # 3. CAR / VEHICLE LOAN
    {
        "policy_id": "POLICY_VEHICLE_LOAN_V1",
        "loan_type": "Car / Vehicle Loan",
        "loan_key": "vehicle_loan",
        "policy_name": "Auto & Commercial Vehicle Finance Policy",
        "version": "1.0",
        "description": "Underwriting policy for new/used four-wheelers and commercial vehicles.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Auto Finance Policy Guidelines 2026",
        "source_section": "Section 5.3 - Passenger Vehicle Underwriting",
        "source_page": 16,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "VL_R01_MIN_INCOME",
                "rule_code": "VL_MIN_INCOME",
                "category": "INCOME",
                "field_name": "monthly_income",
                "operator": "GTE",
                "expected_value": "20000",
                "threshold_value": 20000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum monthly income of ₹20,000 required for vehicle finance.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Auto Finance Policy Guidelines 2026",
                "source_section": "Section 5.3.1 - Income Qualification",
                "source_page": 17
            },
            {
                "rule_id": "VL_R02_MIN_AGE",
                "rule_code": "VL_MIN_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "GTE",
                "expected_value": "21",
                "threshold_value": 21.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Vehicle loan applicant must be at least 21 years old.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Auto Finance Policy Guidelines 2026",
                "source_section": "Section 5.3.2 - Age Parameters",
                "source_page": 18
            },
            {
                "rule_id": "VL_R03_MIN_CIBIL",
                "rule_code": "VL_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "cibil_score",
                "operator": "GTE",
                "expected_value": "650",
                "threshold_value": 650.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum bureau score of 650 required for vehicle finance.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Auto Finance Policy Guidelines 2026",
                "source_section": "Section 5.3.4 - Bureau Thresholds",
                "source_page": 19
            },
            {
                "rule_id": "VL_R04_MAX_LTV",
                "rule_code": "VL_MAX_LTV",
                "category": "COLLATERAL",
                "field_name": "ltv_ratio",
                "operator": "LTE",
                "expected_value": "85",
                "threshold_value": 85.0,
                "severity": "HIGH",
                "mandatory": False,
                "error_message": "Loan-to-Value (LTV) ratio must not exceed 85% on vehicle invoice price.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Auto Finance Policy Guidelines 2026",
                "source_section": "Section 5.4.1 - Loan to Value Limits",
                "source_page": 22
            }
        ],
        "required_documents": [
            {"slot_id": "vl_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "vl_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True},
            {"slot_id": "vl_income", "document_type": "payslip", "display_name": "Income Proof", "required": True},
            {"slot_id": "vl_bank", "document_type": "bank_statement", "display_name": "Bank Statement", "required": True},
            {"slot_id": "vl_quotation", "document_type": "vehicle_quotation", "display_name": "Vehicle Quotation / Proforma Invoice", "required": True}
        ],
        "required_fields": [
            {"document_type": "pan_card", "field_name": "pan_number", "required": True},
            {"document_type": "payslip", "field_name": "net_salary", "required": True},
            {"document_type": "vehicle_quotation", "field_name": "on_road_price", "required": True}
        ]
    },

    # 4. EDUCATION LOAN
    {
        "policy_id": "POLICY_EDUCATION_LOAN_V1",
        "loan_type": "Education Loan",
        "loan_key": "education_loan",
        "policy_name": "Higher Studies Student Credit Policy",
        "version": "1.0",
        "description": "Credit underwriting framework for domestic and overseas higher education courses.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Higher Education Loan Scheme Manual 2026",
        "source_section": "Section 2.1 - Student & Co-Applicant Criteria",
        "source_page": 5,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "EL_R01_MIN_STUDENT_AGE",
                "rule_code": "EL_MIN_STUDENT_AGE",
                "category": "AGE",
                "field_name": "student_age",
                "operator": "GTE",
                "expected_value": "18",
                "threshold_value": 18.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Student applicant must be at least 18 years of age at course admission.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Higher Education Loan Scheme Manual 2026",
                "source_section": "Section 2.1.2 - Student Eligibility",
                "source_page": 6
            },
            {
                "rule_id": "EL_R02_COAPP_MIN_INCOME",
                "rule_code": "EL_COAPP_MIN_INCOME",
                "category": "INCOME",
                "field_name": "coapplicant_income",
                "operator": "GTE",
                "expected_value": "25000",
                "threshold_value": 25000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Co-applicant / Parent monthly net income must be at least ₹25,000.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Higher Education Loan Scheme Manual 2026",
                "source_section": "Section 2.2.1 - Co-Signer Financial Capacity",
                "source_page": 8
            },
            {
                "rule_id": "EL_R03_COAPP_MIN_CIBIL",
                "rule_code": "EL_COAPP_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "coapplicant_cibil",
                "operator": "GTE",
                "expected_value": "650",
                "threshold_value": 650.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Co-applicant bureau score must be 650 or higher.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Higher Education Loan Scheme Manual 2026",
                "source_section": "Section 2.2.3 - Co-Signer Credit Profile",
                "source_page": 10
            },
            {
                "rule_id": "EL_R04_ADMISSION_CONFIRMED",
                "rule_code": "EL_ADMISSION_CONFIRMED",
                "category": "KYC",
                "field_name": "admission_letter",
                "operator": "EXISTS",
                "expected_value": "True",
                "threshold_value": 1.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Valid university admission offer letter is mandatory.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Higher Education Loan Scheme Manual 2026",
                "source_section": "Section 2.3.1 - Academic Documentation",
                "source_page": 11
            }
        ],
        "required_documents": [
            {"slot_id": "el_student_kyc", "document_type": "student_kyc_identity", "display_name": "Student KYC / Identity Proof", "required": True},
            {"slot_id": "el_admission_letter", "document_type": "admission_letter", "display_name": "Admission Letter / Offer Letter", "required": True},
            {"slot_id": "el_fee_structure", "document_type": "fee_structure", "display_name": "Fee Structure / Cost Estimate", "required": True},
            {"slot_id": "el_coapplicant_kyc", "document_type": "coapplicant_kyc_identity", "display_name": "Co-Applicant KYC", "required": True},
            {"slot_id": "el_coapplicant_income", "document_type": "coapplicant_income_proof", "display_name": "Co-Applicant Income Proof", "required": True}
        ],
        "required_fields": [
            {"document_type": "admission_letter", "field_name": "institution_name", "required": True},
            {"document_type": "admission_letter", "field_name": "course_name", "required": True},
            {"document_type": "fee_structure", "field_name": "total_fee", "required": True}
        ]
    },

    # 5. BUSINESS LOAN
    {
        "policy_id": "POLICY_BUSINESS_LOAN_V1",
        "loan_type": "Business Loan",
        "loan_key": "business_loan",
        "policy_name": "MSME & Commercial Business Term Lending Policy",
        "version": "1.0",
        "description": "Credit assessment policy for micro, small, and medium commercial enterprises.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Commercial Lending Guidelines 2026",
        "source_section": "Section 6.1 - SME Credit Standards",
        "source_page": 24,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "BL_R01_MIN_VINTAGE",
                "rule_code": "BL_MIN_VINTAGE",
                "category": "BUSINESS",
                "field_name": "business_vintage_years",
                "operator": "GTE",
                "expected_value": "2",
                "threshold_value": 2.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Business entity must demonstrate at least 2 years of continuous operations.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Commercial Lending Guidelines 2026",
                "source_section": "Section 6.1.1 - Operational Vintage",
                "source_page": 25
            },
            {
                "rule_id": "BL_R02_MIN_TURNOVER",
                "rule_code": "BL_MIN_TURNOVER",
                "category": "INCOME",
                "field_name": "annual_turnover",
                "operator": "GTE",
                "expected_value": "1000000",
                "threshold_value": 1000000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum annual business turnover of ₹10,00,000 required.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Commercial Lending Guidelines 2026",
                "source_section": "Section 6.1.2 - Revenue Thresholds",
                "source_page": 26
            },
            {
                "rule_id": "BL_R03_MIN_CIBIL",
                "rule_code": "BL_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "cibil_score",
                "operator": "GTE",
                "expected_value": "680",
                "threshold_value": 680.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Proprietor / Managing Partner commercial score must be at least 680.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Commercial Lending Guidelines 2026",
                "source_section": "Section 6.2.1 - Commercial Bureau Benchmarks",
                "source_page": 28
            }
        ],
        "required_documents": [
            {"slot_id": "bl_kyc", "document_type": "kyc_identity", "display_name": "Promoter KYC / Identity Proof", "required": True},
            {"slot_id": "bl_pan", "document_type": "pan_card", "display_name": "Business / Promoter PAN", "required": True},
            {"slot_id": "bl_registration", "document_type": "business_registration_proof", "display_name": "Business Registration / Udyam / GST", "required": True},
            {"slot_id": "bl_financials", "document_type": "business_financials", "display_name": "Business Financials / ITR", "required": True},
            {"slot_id": "bl_bank", "document_type": "bank_statement", "display_name": "Current Account Bank Statement", "required": True}
        ],
        "required_fields": [
            {"document_type": "business_registration_proof", "field_name": "registration_number", "required": True},
            {"document_type": "business_financials", "field_name": "gross_receipts", "required": True},
            {"document_type": "bank_statement", "field_name": "account_number", "required": True}
        ]
    },

    # 6. GOLD LOAN
    {
        "policy_id": "POLICY_GOLD_LOAN_V1",
        "loan_type": "Gold Loan",
        "loan_key": "gold_loan",
        "policy_name": "Secured Gold Jewellery Credit Policy",
        "version": "1.0",
        "description": "Short-term demand facility secured by pledge of hallmark gold ornaments.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Bullion & Gold Lending Policy 2026",
        "source_section": "Section 8.1 - Gold Ornament Pledging Guidelines",
        "source_page": 30,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "GL_R01_MIN_AGE",
                "rule_code": "GL_MIN_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "GTE",
                "expected_value": "18",
                "threshold_value": 18.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Pledgor must be an adult aged 18 years or older.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Bullion & Gold Lending Policy 2026",
                "source_section": "Section 8.1.1 - Pledgor Qualifications",
                "source_page": 31
            },
            {
                "rule_id": "GL_R02_MAX_LTV",
                "rule_code": "GL_MAX_LTV",
                "category": "COLLATERAL",
                "field_name": "gold_ltv",
                "operator": "LTE",
                "expected_value": "75",
                "threshold_value": 75.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Regulatory statutory ceiling limits Gold LTV to maximum 75% of appraised value.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Bullion & Gold Lending Policy 2026",
                "source_section": "Section 8.2.3 - Regulatory LTV Ceilings",
                "source_page": 33
            },
            {
                "rule_id": "GL_R03_MIN_PURITY",
                "rule_code": "GL_MIN_PURITY",
                "category": "COLLATERAL",
                "field_name": "gold_karat",
                "operator": "GTE",
                "expected_value": "18",
                "threshold_value": 18.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Pledged gold ornaments must possess certified purity of at least 18 Karats.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Bullion & Gold Lending Policy 2026",
                "source_section": "Section 8.3.1 - Assaying & Purity Standards",
                "source_page": 34
            }
        ],
        "required_documents": [
            {"slot_id": "gl_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "gl_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True}
        ],
        "required_fields": [
            {"document_type": "pan_card", "field_name": "pan_number", "required": True},
            {"document_type": "kyc_identity", "field_name": "name", "required": True}
        ]
    },

    # 7. LOAN AGAINST PROPERTY (LAP)
    {
        "policy_id": "POLICY_LAP_V1",
        "loan_type": "Loan Against Property",
        "loan_key": "loan_against_property",
        "policy_name": "Mortgage Backed Term Lending (LAP) Policy",
        "version": "1.0",
        "description": "Secured facility for personal or business needs against residential or commercial real estate collateral.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Secured Mortgage Underwriting Policy 2026",
        "source_section": "Section 7.1 - Loan Against Property Framework",
        "source_page": 36,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "LAP_R01_MIN_INCOME",
                "rule_code": "LAP_MIN_INCOME",
                "category": "INCOME",
                "field_name": "monthly_income",
                "operator": "GTE",
                "expected_value": "35000",
                "threshold_value": 35000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum monthly net income of ₹35,000 required for Loan Against Property.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Secured Mortgage Underwriting Policy 2026",
                "source_section": "Section 7.1.2 - Borrowing Power Assessment",
                "source_page": 38
            },
            {
                "rule_id": "LAP_R02_MAX_LTV",
                "rule_code": "LAP_MAX_LTV",
                "category": "COLLATERAL",
                "field_name": "lap_ltv",
                "operator": "LTE",
                "expected_value": "65",
                "threshold_value": 65.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "LTV ratio on mortgaged property must not exceed 65% of fair market valuation.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Secured Mortgage Underwriting Policy 2026",
                "source_section": "Section 7.2.1 - Property Valuation & LTV",
                "source_page": 40
            },
            {
                "rule_id": "LAP_R03_MIN_CIBIL",
                "rule_code": "LAP_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "cibil_score",
                "operator": "GTE",
                "expected_value": "675",
                "threshold_value": 675.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum credit score of 675 required for LAP facilities.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Secured Mortgage Underwriting Policy 2026",
                "source_section": "Section 7.3.2 - Credit History & Bureau Cutoffs",
                "source_page": 42
            }
        ],
        "required_documents": [
            {"slot_id": "lap_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "lap_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True},
            {"slot_id": "lap_income", "document_type": "payslip", "display_name": "Income Proof", "required": True},
            {"slot_id": "lap_bank", "document_type": "bank_statement", "display_name": "Bank Statement", "required": True},
            {"slot_id": "lap_property_title", "document_type": "property_title_document", "display_name": "Property Title Deed", "required": True}
        ],
        "required_fields": [
            {"document_type": "property_title_document", "field_name": "property_address", "required": True},
            {"document_type": "property_title_document", "field_name": "owner_name", "required": True}
        ]
    },

    # 8. AGRICULTURE / CROP LOAN
    {
        "policy_id": "POLICY_AGRI_LOAN_V1",
        "loan_type": "Agriculture / Crop Loan",
        "loan_key": "agriculture_loan",
        "policy_name": "Priority Sector Agricultural & Kisan Credit Policy",
        "version": "1.0",
        "description": "Credit line for crop cultivation, farm inputs, and allied agricultural activities.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Agricultural Lending Master Circular 2026",
        "source_section": "Section 9.1 - Kisan Credit Scheme Rules",
        "source_page": 45,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "AG_R01_MIN_LAND_HOLDING",
                "rule_code": "AG_MIN_LAND",
                "category": "COLLATERAL",
                "field_name": "land_acres",
                "operator": "GTE",
                "expected_value": "1.0",
                "threshold_value": 1.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Cultivator must establish legal title or tenancy of at least 1.0 acre of cultivable land.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Agricultural Lending Master Circular 2026",
                "source_section": "Section 9.1.2 - Land Holding Minimums",
                "source_page": 46
            },
            {
                "rule_id": "AG_R02_MIN_AGE",
                "rule_code": "AG_MIN_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "GTE",
                "expected_value": "18",
                "threshold_value": 18.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Primary cultivator must be at least 18 years old.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Agricultural Lending Master Circular 2026",
                "source_section": "Section 9.1.3 - Age Eligibility",
                "source_page": 47
            },
            {
                "rule_id": "AG_R03_CROP_SEASON_VALID",
                "rule_code": "AG_CROP_RECORD",
                "category": "KYC",
                "field_name": "crop_sown",
                "operator": "EXISTS",
                "expected_value": "True",
                "threshold_value": 1.0,
                "severity": "HIGH",
                "mandatory": False,
                "error_message": "Crop inspection or Adangal/Pahani crop declaration required.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Agricultural Lending Master Circular 2026",
                "source_section": "Section 9.2.1 - Seasonal Crop Declarations",
                "source_page": 49
            }
        ],
        "required_documents": [
            {"slot_id": "ag_farmer_kyc", "document_type": "farmer_kyc_identity", "display_name": "Farmer KYC / Identity Proof", "required": True},
            {"slot_id": "ag_pan", "document_type": "pan_card", "display_name": "PAN Card / Form 60", "required": True},
            {"slot_id": "ag_land_records", "document_type": "agricultural_land_records", "display_name": "Agricultural Land Records (7/12, Patta, RTC)", "required": True},
            {"slot_id": "ag_bank", "document_type": "bank_statement", "display_name": "Bank Statement / Passbook", "required": True}
        ],
        "required_fields": [
            {"document_type": "agricultural_land_records", "field_name": "survey_number", "required": True},
            {"document_type": "agricultural_land_records", "field_name": "land_area", "required": True}
        ]
    },

    # 9. LOAN AGAINST FIXED DEPOSIT
    {
        "policy_id": "POLICY_LOAN_AGAINST_FD_V1",
        "loan_type": "Loan Against Fixed Deposit",
        "loan_key": "loan_against_fd",
        "policy_name": "Deposit Collateralized Liquidity Facility Policy",
        "version": "1.0",
        "description": "Instant credit backed by lien marking on unencumbered term deposit receipts.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Term Deposit Credit Scheme Guidelines 2026",
        "source_section": "Section 10.1 - Lien Backed Overdraft Facilities",
        "source_page": 52,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "FD_R01_MIN_DEPOSIT_VALUE",
                "rule_code": "FD_MIN_DEPOSIT",
                "category": "COLLATERAL",
                "field_name": "fd_principal_amount",
                "operator": "GTE",
                "expected_value": "25000",
                "threshold_value": 25000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Collateral term deposit principal must be at least ₹25,000.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Term Deposit Credit Scheme Guidelines 2026",
                "source_section": "Section 10.1.2 - Minimum Deposit Amount",
                "source_page": 53
            },
            {
                "rule_id": "FD_R02_MAX_LTV",
                "rule_code": "FD_MAX_LTV",
                "category": "COLLATERAL",
                "field_name": "fd_ltv",
                "operator": "LTE",
                "expected_value": "90",
                "threshold_value": 90.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Maximum advance value is capped at 90% of principal plus accrued interest.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Term Deposit Credit Scheme Guidelines 2026",
                "source_section": "Section 10.2.1 - Collateral Haircut & LTV",
                "source_page": 54
            },
            {
                "rule_id": "FD_R03_MIN_TENOR_LEFT",
                "rule_code": "FD_MIN_TENOR",
                "category": "COLLATERAL",
                "field_name": "fd_tenor_months_remaining",
                "operator": "GTE",
                "expected_value": "3",
                "threshold_value": 3.0,
                "severity": "HIGH",
                "mandatory": True,
                "error_message": "Underlying fixed deposit must have at least 3 months residual maturity.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Term Deposit Credit Scheme Guidelines 2026",
                "source_section": "Section 10.2.4 - Maturity Matching",
                "source_page": 55
            }
        ],
        "required_documents": [
            {"slot_id": "fd_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "fd_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True},
            {"slot_id": "fd_receipt", "document_type": "fixed_deposit_receipt", "display_name": "Fixed Deposit Receipt / Certificate", "required": True}
        ],
        "required_fields": [
            {"document_type": "fixed_deposit_receipt", "field_name": "deposit_amount", "required": True},
            {"document_type": "fixed_deposit_receipt", "field_name": "fd_number", "required": True}
        ]
    },

    # 10. CONSUMER DURABLE LOAN
    {
        "policy_id": "POLICY_CONSUMER_DURABLE_V1",
        "loan_type": "Consumer Durable Loan",
        "loan_key": "consumer_durable_loan",
        "policy_name": "Point-of-Sale Consumer Electronics Credit Policy",
        "version": "1.0",
        "description": "Retail installment finance for electronics, household appliances, and gadgets.",
        "effective_date": "2026-01-01",
        "source_type": "DEMO_POLICY",
        "source_document": "Demo Consumer POS Credit Policy 2026",
        "source_section": "Section 11.2 - Durable Purchase Eligibility",
        "source_page": 58,
        "status": "ACTIVE",
        "rules": [
            {
                "rule_id": "CD_R01_MIN_INCOME",
                "rule_code": "CD_MIN_INCOME",
                "category": "INCOME",
                "field_name": "monthly_income",
                "operator": "GTE",
                "expected_value": "15000",
                "threshold_value": 15000.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum monthly net income of ₹15,000 required for consumer durable financing.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Consumer POS Credit Policy 2026",
                "source_section": "Section 11.2.1 - Income Qualification",
                "source_page": 59
            },
            {
                "rule_id": "CD_R02_MIN_AGE",
                "rule_code": "CD_MIN_AGE",
                "category": "AGE",
                "field_name": "applicant_age",
                "operator": "GTE",
                "expected_value": "21",
                "threshold_value": 21.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Applicant must be at least 21 years of age.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Consumer POS Credit Policy 2026",
                "source_section": "Section 11.2.2 - Age Parameters",
                "source_page": 60
            },
            {
                "rule_id": "CD_R03_MIN_CIBIL",
                "rule_code": "CD_MIN_CIBIL",
                "category": "CREDIT",
                "field_name": "cibil_score",
                "operator": "GTE",
                "expected_value": "650",
                "threshold_value": 650.0,
                "severity": "CRITICAL",
                "mandatory": True,
                "error_message": "Minimum credit bureau score of 650 is required.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Consumer POS Credit Policy 2026",
                "source_section": "Section 11.2.4 - Score Cutoffs",
                "source_page": 61
            },
            {
                "rule_id": "CD_R04_INVOICE_EXISTS",
                "rule_code": "CD_INVOICE_CHECK",
                "category": "COLLATERAL",
                "field_name": "product_invoice",
                "operator": "EXISTS",
                "expected_value": "True",
                "threshold_value": 1.0,
                "severity": "HIGH",
                "mandatory": False,
                "error_message": "Merchant proforma invoice or quotation required for loan disbursement.",
                "source_type": "DEMO_POLICY",
                "source_document": "Demo Consumer POS Credit Policy 2026",
                "source_section": "Section 11.3.1 - Asset Documentation",
                "source_page": 63
            }
        ],
        "required_documents": [
            {"slot_id": "cd_kyc", "document_type": "kyc_identity", "display_name": "KYC / Identity Proof", "required": True},
            {"slot_id": "cd_pan", "document_type": "pan_card", "display_name": "PAN Card", "required": True},
            {"slot_id": "cd_income", "document_type": "payslip", "display_name": "Income Proof", "required": True},
            {"slot_id": "cd_bank", "document_type": "bank_statement", "display_name": "Bank Statement", "required": True},
            {"slot_id": "cd_quotation", "document_type": "product_invoice", "display_name": "Product Invoice / Quotation", "required": False}
        ],
        "required_fields": [
            {"document_type": "pan_card", "field_name": "pan_number", "required": True},
            {"document_type": "payslip", "field_name": "net_salary", "required": True}
        ]
    }
]


def get_policy_definition(loan_type: str) -> Optional[Dict[str, Any]]:
    """Retrieves policy dictionary by loan type name or key."""
    norm = loan_type.strip().lower().replace(" ", "_").replace("/", "_")
    for pol in ALL_LOAN_POLICIES:
        pol_key = pol["loan_key"].strip().lower()
        pol_name = pol["loan_type"].strip().lower().replace(" ", "_").replace("/", "_")
        if norm == pol_key or norm == pol_name:
            return pol
    # Fallback partial matching
    for pol in ALL_LOAN_POLICIES:
        if pol["loan_key"] in norm or norm in pol["loan_key"]:
            return pol
    return None
