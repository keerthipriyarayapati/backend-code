"""
Synthetic Dataset Generator for Loan Document AI Agent 1 Evaluation.
Generates completely synthetic documents with zero real PII across all 9 classes and 8 file formats.
"""

import os
import sys
from pathlib import Path
import pymupdf
import docx
from PIL import Image, ImageDraw, ImageFont

DATASET_DIR = Path(__file__).parent / "dataset_files"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 45 Fictional Synthetic Samples (5 per category across various file formats)
SYNTHETIC_SAMPLES_CONFIG = [
    # 1. PAYSLIP SAMPLES
    {
        "filename": "payslip_01.pdf",
        "ground_truth": "payslip",
        "format": "pdf",
        "text": """ACME GLOBAL SOLUTIONS INC.
MONTHLY PAYSLIP - JUNE 2025
Employee Name: Fictional Employee 1 | ID: EMP-1002
Basic Salary: $5,000.00
House Rent Allowance (HRA): $2,000.00
Special Allowance: $1,500.00
Gross Salary: $8,500.00
Provident Fund (PF): $600.00 | Income Tax (TDS): $900.00
Total Deductions: $1,500.00
NET PAYABLE SALARY: $7,000.00
Direct Credited to Employee Account."""
    },
    {
        "filename": "payslip_02.docx",
        "ground_truth": "payslip",
        "format": "docx",
        "text": """NEXUS TECH ENTERPRISES
SALARY WAGE SLIP - MAY 2025
Employee: Jane Fictional | Designation: Senior Engineer
Basic Pay: 45,000 INR
Dearness Allowance: 10,000 INR
Medical Allowance: 5,000 INR
Gross Earnings: 60,000 INR
Professional Tax: 200 INR | PF: 1,800 INR
Net Pay Transferred: 58,000 INR"""
    },
    {
        "filename": "payslip_03.txt",
        "ground_truth": "payslip",
        "format": "txt",
        "text": """GLOBAL CORP PAYROLL STATEMENT
Period: April 2025
Salary breakdown:
Base Wage: $4,200
Bonus & Incentive: $800
Gross Compensation: $5,000
Tax Deducted: $500
Net Disbursement: $4,500"""
    },
    {
        "filename": "payslip_tricky_with_bank_info.pdf",
        "ground_truth": "payslip",
        "format": "pdf",
        "text": """OMEGA LOGISTICS PVT LTD
SALARY ADVICE NOTE FOR JULY 2025
Employee: Alex Fictional | Code: OMEGA-4421
Basic Pay: 65,000 INR | Allowances: 25,000 INR
Deductions: 8,000 INR | Net Salary: 82,000 INR
Salary Payment Method: Direct Bank Transfer to State Fictional Bank Account 987654321012."""
    },
    {
        "filename": "payslip_05.png",
        "ground_truth": "payslip",
        "format": "png",
        "text": "PAYROLL SLIP - AUGUST 2025\nBasic: $3000\nNet Salary: $2700"
    },

    # 2. BANK STATEMENT SAMPLES
    {
        "filename": "bank_statement_01.pdf",
        "ground_truth": "bank_statement",
        "format": "pdf",
        "text": """NATIONAL FICTIONAL BANK
ACCOUNT STATEMENT FOR ACCOUNT NO: 112233445566
Statement Period: 01-Jan-2025 to 31-Jan-2025
Opening Balance: $12,450.00
Date        Description           Debit ($)    Credit ($)   Balance ($)
02-Jan-2025 ATM Withdrawal        200.00                    12,250.00
05-Jan-2025 Grocery Store Purchase 150.00                    12,100.00
15-Jan-2025 Utility Bill Payment  100.00                    12,000.00
31-Jan-2025 Total Monthly Interest             25.00        12,025.00
Closing Balance: $12,025.00"""
    },
    {
        "filename": "bank_statement_tricky_salary_credit.pdf",
        "ground_truth": "bank_statement",
        "format": "pdf",
        "text": """METRO COMMUNITY BANK
SAVINGS ACCOUNT STATEMENT
Account Number: 5544332211
Customer: Alex Customer
01-Jun-2025 Opening Balance: 50,000 INR
05-Jun-2025 ACH Credit - SALARY CREDIT FROM ACME CORP: +75,000 INR | Balance: 125,000 INR
10-Jun-2025 POS Debit - Supermarket: -3,000 INR | Balance: 122,000 INR
20-Jun-2025 NEFT Outward Transfer: -15,000 INR | Balance: 107,000 INR
30-Jun-2025 Closing Ledger Balance: 107,000 INR"""
    },
    {
        "filename": "bank_statement_03.docx",
        "ground_truth": "bank_statement",
        "format": "docx",
        "text": """FIRST TRUST BANKING STATEMENT
Account: 998877665544
Transactions:
Debit: $450 - Rent Transfer
Credit: $1200 - Customer Deposit
Closing Available Ledger Balance: $4,580.00"""
    },
    {
        "filename": "bank_statement_04.txt",
        "ground_truth": "bank_statement",
        "format": "txt",
        "text": """BANK STATEMENT SUMMARY
Acct No: 88776655
Period: Q1 2025
Total Credits: $15,000
Total Debits: $11,200
Ending Balance: $3,800"""
    },
    {
        "filename": "bank_statement_05.svg",
        "ground_truth": "bank_statement",
        "format": "svg",
        "text": "BANK STATEMENT\nAccount Balance: $10,500\nRecent Transactions: Debit $500, Credit $200"
    },

    # 3. ITR / TAX RETURN SAMPLES
    {
        "filename": "itr_tax_return_01.pdf",
        "ground_truth": "itr_tax_return",
        "format": "pdf",
        "text": """INCOME TAX DEPARTMENT
ACKNOWLEDGEMENT OF INDIAN INCOME TAX RETURN
Assessment Year: 2024-2025 | Financial Year: 2023-2024
Form Type: ITR-1 (SAHAJ)
Name: Fictional Taxpayer
Gross Total Income: 1,200,000 INR
Total Deductions under Chapter VI-A (80C, 80D): 200,000 INR
Total Taxable Income: 1,000,000 INR
Net Tax Payable: 112,500 INR
TDS Paid: 112,500 INR | Refund Due: 0 INR
E-Filing Verification Code: 987654321012"""
    },
    {
        "filename": "itr_tax_return_02.docx",
        "ground_truth": "itr_tax_return",
        "format": "docx",
        "text": """FEDERAL TAX RETURN ACKNOWLEDGEMENT
Form 1040 - US Individual Income Tax Return
Tax Year: 2024
Filer Name: John Fictional Taxpayer
Adjusted Gross Income (AGI): $85,000
Standard Deduction: $13,850
Taxable Income: $71,150
Total Federal Income Tax Withheld: $12,400"""
    },
    {
        "filename": "itr_tax_return_03.txt",
        "ground_truth": "itr_tax_return",
        "format": "txt",
        "text": """INCOME TAX RETURN FILING SUMMARY
Assessment Year 2025-2026
Gross Income: 950,000 INR
Tax Paid: 75,000 INR
Filing Status: Successfully Verified electronically."""
    },
    {
        "filename": "itr_tax_return_04.pdf",
        "ground_truth": "itr_tax_return",
        "format": "pdf",
        "text": """STATE TAX AUTHORITIES
ANNUAL TAX FILING REPORT
Assessment Year: 2024
Total Computation of Taxable Profits: $110,000
Net Tax Paid under Section 143(1): $14,500"""
    },
    {
        "filename": "itr_tax_return_05.png",
        "ground_truth": "itr_tax_return",
        "format": "png",
        "text": "INCOME TAX RETURN\nAssessment Year: 2024-2025\nTaxable Income: $50000\nTax Payable: $5000"
    },

    # 4. KYC / IDENTITY SAMPLES
    {
        "filename": "kyc_identity_01.pdf",
        "ground_truth": "kyc_identity",
        "format": "pdf",
        "text": """GOVERNMENT OF INDIA IDENTITY DOCUMENT
PERMANENT ACCOUNT NUMBER (PAN CARD)
Name: FICTIONAL CITIZEN
Father's Name: PARENT CITIZEN
Date of Birth: 15/08/1990
PAN Number: ABCDE1234F
Signature & Government Photo Identity Verified."""
    },
    {
        "filename": "kyc_identity_02.png",
        "ground_truth": "kyc_identity",
        "format": "png",
        "text": "GOVERNMENT IDENTITY CARD\nAadhaar No: 9999 8888 7777\nDOB: 01/01/1995\nAddress: 123 Sample Street"
    },
    {
        "filename": "kyc_identity_03.docx",
        "ground_truth": "kyc_identity",
        "format": "docx",
        "text": """PASSPORT OFFICE - GOVERNMENT IDENTITY EXPORT
Document: Republic Passport
Passport No: Z9876543
Full Name: FICTIONAL TRAVELER
Nationality: Fictionalland
Date of Birth: 24-NOV-1988
Place of Birth: Metropolitan City"""
    },
    {
        "filename": "kyc_identity_04.txt",
        "ground_truth": "kyc_identity",
        "format": "txt",
        "text": """NATIONAL DRIVER LICENSE IDENTIFICATION
License No: DL-1420110099887
Name: FICTIONAL DRIVER
Date of Issue: 10/10/2020
Expiry Date: 09/10/2040
Identity Photo & Signature on file."""
    },
    {
        "filename": "kyc_identity_05.svg",
        "ground_truth": "kyc_identity",
        "format": "svg",
        "text": "VOTER IDENTITY CARD\nID No: ABC1234567\nElector Name: Fictional Voter\nDOB: 12/12/1992"
    },

    # 5. EMPLOYMENT LETTER SAMPLES
    {
        "filename": "employment_letter_01.pdf",
        "ground_truth": "employment_letter",
        "format": "pdf",
        "text": """ACME INNOVATIONS LTD.
OFFER OF EMPLOYMENT & APPOINTMENT LETTER
Date: March 15, 2025
Dear Candidate,
We are pleased to offer you employment at ACME Innovations Ltd as Software Architect.
Your joining date will be April 1, 2025.
This position is full-time employment based in our Corporate Headquarters.
Sincerely,
Human Resources Director"""
    },
    {
        "filename": "employment_letter_tricky_salary_mention.docx",
        "ground_truth": "employment_letter",
        "format": "docx",
        "text": """GLOBAL TECH CORP HR DEPARTMENT
CONFIRMATION OF EMPLOYMENT LETTER
To Whom It May Concern,
This letter confirms that Fictional Employee has been employed with Global Tech Corp since January 10, 2022 as Senior Project Manager.
The employee is currently on an annual salary compensation of $95,000 per year.
This letter is issued upon request of the employee for verification purposes."""
    },
    {
        "filename": "employment_letter_03.txt",
        "ground_truth": "employment_letter",
        "format": "txt",
        "text": """JOB APPOINTMENT LETTER
Company: Apex Solutions
Designation: Operations Manager
Start Date: May 1, 2025
Employment Status: Permanent / Full-time"""
    },
    {
        "filename": "employment_letter_04.pdf",
        "ground_truth": "employment_letter",
        "format": "pdf",
        "text": """HR DEPT CERTIFICATE OF SERVICE
This is to certify that Mr. Sample Engineer is currently working in our organization as Lead Systems Engineer since September 2021."""
    },
    {
        "filename": "employment_letter_05.png",
        "ground_truth": "employment_letter",
        "format": "png",
        "text": "EMPLOYMENT VERIFICATION LETTER\nEmployee: Jane Doe\nRole: Lead Designer\nStatus: Active Staff"
    },

    # 6. FORM 16 SAMPLES
    {
        "filename": "form_16_01.pdf",
        "ground_truth": "form_16",
        "format": "pdf",
        "text": """FORM NO. 16
[See rule 31(1)(a)]
Certificate under section 203 of the Income-tax Act, 1961 for tax deducted at source from income chargeable under the head "Salaries".
PART A & PART B
Name and address of Employer: ACME ENTERPRISES TAN: DELA12345B
Name and address of Employee: FICTIONAL WORKER PAN: ABCDE9999Z
Assessment Year: 2024-2025
Gross Salary: 1,500,000 INR
Total Deductions under Chapter VI-A: 150,000 INR
Tax Deducted at Source (TDS): 180,000 INR
TDS Deposited with Central Government Account."""
    },
    {
        "filename": "form_16_02.docx",
        "ground_truth": "form_16",
        "format": "docx",
        "text": """FORM 16 CERTIFICATE OF TDS ON SALARY
Assessment Year: 2025-2026
Employer TAN: MUMK99887A
Employee PAN: XYZP11223K
Total Salary Paid: 980,000 INR
Tax Deducted under Section 192: 85,000 INR
Quarterly TDS Deposit Details Verified."""
    },
    {
        "filename": "form_16_03.txt",
        "ground_truth": "form_16",
        "format": "txt",
        "text": """FORM 16 PART B DEDUCTION DETAILS
Assessment Year 2024-25
Gross Compensation: 800,000 INR
Tax Payable: 50,000 INR
Total Tax Deducted at Source (TDS): 50,000 INR"""
    },
    {
        "filename": "form_16_04.pdf",
        "ground_truth": "form_16",
        "format": "pdf",
        "text": """INCOME TAX FORM 16 SALARY CERTIFICATE
Employer Name: Tech Solutions Ltd
Employee Name: Test Employee
TDS Certificate No: FORM16-2024-8849
Assessment Year: 2024-2025"""
    },
    {
        "filename": "form_16_05.svg",
        "ground_truth": "form_16",
        "format": "svg",
        "text": "FORM 16 CERTIFICATE\nAssessment Year: 2024-2025\nTotal Salary: 750,000 INR\nTotal TDS Deducted: 60,000 INR"
    },

    # 7. ADDRESS PROOF SAMPLES
    {
        "filename": "address_proof_01.pdf",
        "ground_truth": "address_proof",
        "format": "pdf",
        "text": """CITY ELECTRICITY DISTRIBUTION POWER CO.
MONTHLY ELECTRICITY UTILITY BILL
Consumer No: 987654321
Consumer Name: FICTIONAL RESIDENT
Billing Address: Plot 42, Fictional Colony, Sector 15, City 400001
Bill Date: 10-Jun-2025
Units Consumed: 240 kWh
Total Amount Due: $85.00
This bill serves as a valid residential address proof document."""
    },
    {
        "filename": "address_proof_tricky_identity_info.docx",
        "ground_truth": "address_proof",
        "format": "docx",
        "text": """RESIDENTIAL LEASE TENANCY AGREEMENT
This agreement made on May 1, 2025 between Landlord Fictional and Tenant Fictional (Passport ID Z9876543).
Premises Address: Apartment 4B, Sunshine Towers, Ocean Avenue, Fictional City 10001.
Tenant shall occupy the premises as primary residence for 12 months."""
    },
    {
        "filename": "address_proof_03.txt",
        "ground_truth": "address_proof",
        "format": "txt",
        "text": """MUNICIPAL WATER SUPPLY BILL
Consumer: Residence Owner
Service Address: 742 Evergreen Terrace, Sector 7
Billing Period: Q2 2025
Current Charges: $45.00
Residential verification address confirmed."""
    },
    {
        "filename": "address_proof_04.pdf",
        "ground_truth": "address_proof",
        "format": "pdf",
        "text": """TELECOM FIBER BROADBAND UTILITY BILL
Subscriber Name: Customer Resident
Installation Address: Suite 300, Innovation Way, Tech Park 500081
Statement Month: May 2025"""
    },
    {
        "filename": "address_proof_05.png",
        "ground_truth": "address_proof",
        "format": "png",
        "text": "UTILITY BILL - GAS SERVICE\nCustomer: John Smith\nService Address: 100 Park Place\nAmount: $62"
    },

    # 8. OTHER SAMPLES (Readable non-loan docs)
    {
        "filename": "other_01.pdf",
        "ground_truth": "other",
        "format": "pdf",
        "text": """USER MANUAL - SMART HOME TEMPERATURE CONTROLLER
Chapter 1: Installation Instructions
1. Mount the thermostat unit on an interior wall.
2. Connect red wire to R terminal and white wire to W terminal.
3. Turn on the circuit breaker and pair with mobile app."""
    },
    {
        "filename": "other_02.docx",
        "ground_truth": "other",
        "format": "docx",
        "text": """RECIPE BOOK: ITALIAN PASTA DISHES
Ingredients:
- 250g Spaghetti
- 2 tablespoons olive oil
- 3 cloves garlic, minced
Instructions: Boil pasta in salted water for 9 minutes. Saute garlic in olive oil."""
    },
    {
        "filename": "other_03.txt",
        "ground_truth": "other",
        "format": "txt",
        "text": """CONFERENCE MEETING MINUTES
Project Alpha Architecture Discussion
Attendees: Alice, Bob, Charlie
Decisions made: Migrate microservice to event-driven queue."""
    },
    {
        "filename": "other_04.pdf",
        "ground_truth": "other",
        "format": "pdf",
        "text": """UNIVERSITY LECTURE NOTES - ALGORITHMS 101
Topic: Binary Search Trees
Time Complexity: O(log N) average case, O(N) worst case."""
    },
    {
        "filename": "other_05.svg",
        "ground_truth": "other",
        "format": "svg",
        "text": "PROJECT TIMELINE GANTT CHART\nMilestone 1: Prototype\nMilestone 2: Release"
    },

    # 9. UNKNOWN SAMPLES (Unreadable, corrupted, empty, or ambiguous)
    {
        "filename": "unknown_empty.txt",
        "ground_truth": "unknown",
        "format": "txt",
        "text": ""  # Empty text
    },
    {
        "filename": "unknown_short_noise.txt",
        "ground_truth": "unknown",
        "format": "txt",
        "text": "asdf qwer zxcv 1234 !!!"
    },
    {
        "filename": "unknown_corrupted.pdf",
        "ground_truth": "unknown",
        "format": "pdf",
        "corrupted": True
    },
    {
        "filename": "unknown_encrypted.pdf",
        "ground_truth": "unknown",
        "format": "pdf",
        "encrypted": True
    },
    {
        "filename": "unknown_ambiguous.txt",
        "ground_truth": "unknown",
        "format": "txt",
        "text": "DOCUMENT SUMMARY:\nTotal amount: 5000\nDate: 2025-01-01\nStatus: Pending review"
    }
]


def create_pdf_file(filepath: Path, text: str, encrypted: bool = False, corrupted: bool = False):
    """Generates synthetic PDF files using PyMuPDF."""
    if corrupted:
        with open(filepath, "wb") as f:
            f.write(b"%PDF-1.4 CORRUPTED FILE DATA NOT A VALID PDF STRUCTURE")
        return

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=11)

    if encrypted:
        # Save with password encryption
        doc.save(str(filepath), encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="admin", user_pw="secret")
    else:
        doc.save(str(filepath))
    doc.close()


def create_docx_file(filepath: Path, text: str):
    """Generates synthetic DOCX files using python-docx."""
    doc = docx.Document()
    for line in text.split("\n"):
        doc.add_paragraph(line)
    doc.save(str(filepath))


def create_txt_file(filepath: Path, text: str):
    """Generates synthetic TXT files."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def create_png_file(filepath: Path, text: str):
    """Generates synthetic PNG image files using Pillow."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text or "[EMPTY IMAGE]", fill=(0, 0, 0))
    img.save(str(filepath), format="PNG")


def create_svg_file(filepath: Path, text: str):
    """Generates synthetic SVG files."""
    escaped_lines = "\n".join(f'<text x="20" y="{40 + i*30}">{line}</text>' for i, line in enumerate((text or "EMPTY").split("\n")))
    svg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="600" height="400" xmlns="http://www.w3.org/2000/svg">
  <rect width="100%" height="100%" fill="#ffffff"/>
  {escaped_lines}
</svg>"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(svg_content)


def generate_synthetic_dataset():
    """Generates all synthetic dataset files on disk."""
    generated_files = []

    for item in SYNTHETIC_SAMPLES_CONFIG:
        fname = item["filename"]
        fpath = DATASET_DIR / fname
        fmt = item["format"]
        text = item.get("text", "")
        encrypted = item.get("encrypted", False)
        corrupted = item.get("corrupted", False)

        if fmt == "pdf":
            create_pdf_file(fpath, text, encrypted=encrypted, corrupted=corrupted)
        elif fmt == "docx":
            create_docx_file(fpath, text)
        elif fmt == "txt":
            create_txt_file(fpath, text)
        elif fmt in ["png", "jpg", "jpeg"]:
            create_png_file(fpath, text)
        elif fmt == "svg":
            create_svg_file(fpath, text)

        generated_files.append({
            "filename": fname,
            "file_path": str(fpath.resolve()),
            "ground_truth": item["ground_truth"],
            "file_extension": fpath.suffix.lower()
        })

    print(f"Generated {len(generated_files)} synthetic evaluation files in {DATASET_DIR}")
    return generated_files


def get_synthetic_evaluation_dataset():
    """Returns dataset configuration array, generating files if not already present."""
    if not (DATASET_DIR / "payslip_01.pdf").exists():
        return generate_synthetic_dataset()

    dataset = []
    for item in SYNTHETIC_SAMPLES_CONFIG:
        fname = item["filename"]
        fpath = DATASET_DIR / fname
        if not fpath.exists():
            generate_synthetic_dataset()
            break

    dataset = []
    for item in SYNTHETIC_SAMPLES_CONFIG:
        fname = item["filename"]
        fpath = DATASET_DIR / fname
        dataset.append({
            "filename": fname,
            "file_path": str(fpath.resolve()),
            "ground_truth": item["ground_truth"],
            "file_extension": fpath.suffix.lower()
        })
    return dataset


# Auto-generate on module load to guarantee files exist for tests
get_synthetic_evaluation_dataset()

