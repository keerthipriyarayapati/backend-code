"""
End-to-End Test Script for Education Loan Multi-Agent Pipeline Fixes.
Verifies all 10 user requirements across Agents 1 to 6.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.policy import get_loan_type_policy
from agents.agent_1_document.agent import DocumentClassificationAgent
from agents.agent_2_extraction.agent import InformationExtractionAgent
from agents.agent_3_validation.agent import InformationValidationAgent
from agents.agent_4_cross_document.agent import CrossDocumentVerifier, resolve_document_role
from agents.agent_5_risk.agent import RiskAnomalyDetectionAgent
from agents.agent_6_report.agent import FinalReportDecisionAgent


def test_education_loan_pipeline_fixes():
    print("=" * 80)
    print("EXECUTING EDUCATION LOAN MULTI-AGENT PIPELINE E2E TEST")
    print("=" * 80)

    # 1. Initialize Agents
    agent_1 = DocumentClassificationAgent()
    agent_2 = InformationExtractionAgent()
    agent_3 = InformationValidationAgent()
    agent_4 = CrossDocumentVerifier()
    agent_5 = RiskAnomalyDetectionAgent()
    agent_6 = FinalReportDecisionAgent()

    # 2. Define synthetic test documents for the 6 required slots
    # Slot 1: Student KYC (Intentional missing identity_document_type)
    student_kyc_text = """STUDENT IDENTITY PROOF / KYC
Full Name: Choudam Bharath Reddy
Date of Birth: 15-08-2002
Gender: Male
Identity Document Number: 987654321012
Address: 123 University Campus Road, Hyderabad
(identity_document_type intentionally omitted to test missing field)"""

    # Slot 2: Admission Letter
    admission_letter_text = """ADMISSION OFFER LETTER
Student Name: Choudam Bharath Reddy
Institution Name: National Institute of Technology, Hyderabad
Course Name: Master of Technology in AI
Admission Number: NITH-MTECH-2026-904
Admission Date: 10-06-2026
Academic Year: 2026-2028
Status: Admitted"""

    # Slot 3: Fee Structure
    fee_structure_text = """OFFICIAL UNIVERSITY FEE STRUCTURE
Student Name: Choudam Bharath Reddy
Institution Name: National Institute of Technology, Hyderabad
Course Name: Master of Technology in AI
Academic Year: 2026-2028
Tuition Fee: INR 150000
Hostel Fee: INR 50000
Examination Fee: INR 5000
Total Fee: INR 205000
Issue Date: 15-06-2026"""

    # Slot 4: Academic Certificate
    academic_certificate_text = """BOARD OF TECHNICAL EDUCATION
PROVISIONAL DEGREE CERTIFICATE
Student Name: Choudam Bharath Reddy
Institution Name: National Institute of Technology, Hyderabad
Examination: Bachelor of Technology
Roll Number: NITH-BTECH-2022-401
Academic Year: 2022-2026
Marks Obtained: 850
Total Marks: 1000
Percentage: 85%
Result: Pass"""

    # Slot 5: Co-applicant KYC
    coapp_kyc_text = """CO-APPLICANT PAN CARD
Full Name: C Hanumantha Reddy
Name: C Hanumantha Reddy
Identity Document Type: PAN
Identity Document Number: ABCDE1234F
PAN: ABCDE1234F
Date of Birth: 10-05-1975
Gender: Male"""

    # Slot 6: Co-applicant Income Proof (Payslip)
    coapp_income_text = """PAYSLIP FOR MONTH OF JULY 2026
Employee Name: C Hanumantha Reddy
Employer Name: Tech Solutions Ltd
Designation: Senior Manager
Pay Period: July 2026
Basic Salary: INR 60000
Gross Salary: INR 90000
Net Salary: INR 78000
Net Payable Salary: INR 78000
Deductions: INR 12000"""

    scratch_dir = PROJECT_ROOT / "scratch" / "test_edu_files"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    files_info = [
        ("01_student_kyc.txt", student_kyc_text, "edu_loan_student_kyc", "STUDENT"),
        ("02_admission_letter.txt", admission_letter_text, "edu_loan_admission_letter", "STUDENT"),
        ("03_fee_structure.txt", fee_structure_text, "edu_loan_fee_structure", "STUDENT"),
        ("04_academic_certificate.txt", academic_certificate_text, "edu_loan_academic_proof", "STUDENT"),
        ("05_coapp_kyc.txt", coapp_kyc_text, "edu_loan_co_app_kyc", "CO_APPLICANT"),
        ("06_coapp_income.txt", coapp_income_text, "edu_loan_co_app_income", "CO_APPLICANT"),
    ]

    file_paths = []
    slot_map = {}
    for filename, text, slot_id, role in files_info:
        p = scratch_dir / filename
        with open(p, "w", encoding="utf-8") as f:
            f.write(text)
        file_paths.append(str(p))
        slot_map[str(p)] = (slot_id, role)

    # STEP A: AGENT 1 CLASSIFICATION
    c_results = agent_1.process_batch(file_paths)
    print(f"\n[AGENT 1] Classified {len(c_results)} documents.")
    for cr in c_results:
        f_name = Path(cr.get("filename", "")).name
        dt = cr.get("document_type")
        for p_key, (s_id, r_val) in slot_map.items():
            if f_name in p_key:
                cr["requirement_id"] = s_id
                cr["entity_role"] = r_val
                cr["document_role"] = r_val
        print(f"  - {f_name} -> Classified as '{dt}' (Role: {cr.get('entity_role')})")

    # STEP B: AGENT 2 EXTRACTION
    e_results = agent_2.process_batch(c_results)
    print(f"\n[AGENT 2] Extracted data for {len(e_results)} documents.")
    
    # Assert Academic Certificate institution_name extracted properly
    acad_doc = next(e for e in e_results if e.get("document_type") in ["academic_certificate", "marksheet"])
    acad_inst = acad_doc.get("fields", {}).get("institution_name", {}).get("value")
    print(f"  - Academic Certificate institution_name: '{acad_inst}'")
    assert acad_inst == "National Institute of Technology, Hyderabad", f"Expected institution_name extracted, got '{acad_inst}'"

    # Assert Student KYC identity_document_type is None
    stud_doc = next(e for e in e_results if e.get("requirement_id") == "edu_loan_student_kyc" or "student_kyc" in str(e))
    id_type = stud_doc.get("fields", {}).get("identity_document_type", {}).get("value")
    print(f"  - Student KYC identity_document_type: {id_type}")
    assert id_type is None, f"Expected identity_document_type to be None, got '{id_type}'"

    # STEP C: AGENT 3 VALIDATION
    v_results = agent_3.process_batch(e_results)
    print(f"\n[AGENT 3] Validated {len(v_results)} documents.")

    tot_val_errors = 0
    stud_kyc_val_errors = 0
    for vr in v_results:
        dt = vr.get("document_type")
        status = vr.get("validation_status")
        errs = vr.get("error_count", 0)
        tot_val_errors += errs
        if vr.get("requirement_id") == "edu_loan_student_kyc" or dt == "kyc_identity":
            stud_kyc_val_errors = errs
        print(f"  - {dt}: Status={status}, Errors={errs}")

    assert stud_kyc_val_errors == 1, f"Expected exactly 1 validation error on Student KYC, got {stud_kyc_val_errors}"
    assert tot_val_errors == 1, f"Expected total 1 validation error across all docs, got {tot_val_errors}"

    # STEP D: AGENT 4 CROSS-DOCUMENT VERIFICATION
    for er in e_results:
        for p_key, (s_id, r_val) in slot_map.items():
            if Path(er.get("filename", "")).name in p_key:
                er["requirement_id"] = s_id
                er["entity_role"] = r_val
                er["document_role"] = r_val

    cross_res = agent_4.process(e_results, loan_type="education_loan", application_id="APP-EDU-FIX-TEST")
    print(f"\n[AGENT 4] Cross-Document Verification completed.")
    print(f"  - Coverage: {cross_res.verification_coverage:.1f}%")
    print(f"  - Matches: {cross_res.match_count}")
    print(f"  - Mismatches: {cross_res.mismatch_count}")
    print(f"  - Role Separation (NOT_APPLICABLE): {sum(1 for f in cross_res.findings if f.comparison_result == 'NOT_APPLICABLE')}")

    assert cross_res.mismatch_count == 0, f"Expected 0 cross-document mismatches, got {cross_res.mismatch_count}"

    # STEP E: AGENT 5 RISK ASSESSMENT
    state_input = {
        "application_id": "APP-EDU-FIX-TEST",
        "loan_type": "education_loan",
        "classification_results": c_results,
        "extraction_results": e_results,
        "validation_results": v_results,
        "cross_document_results": cross_res.model_dump() if hasattr(cross_res, "model_dump") else cross_res,
        "_has_classification_input": True
    }
    
    risk_res = agent_5._graph.invoke(state_input).get("risk_assessment_results")
    if hasattr(risk_res, "model_dump"):
        risk_data = risk_res.model_dump()
    elif isinstance(risk_res, dict):
        risk_data = risk_res
    else:
        risk_data = {}

    r_score = risk_data.get("risk_score", 0.0)
    r_level = risk_data.get("risk_level")
    r_factors = risk_data.get("risk_factors") or []
    print(f"\n[AGENT 5] Risk Assessment completed.")
    print(f"  - Risk Score: {r_score:.1f}/100")
    print(f"  - Risk Level: {r_level}")
    print(f"  - Active Risk Factors ({len(r_factors)}):")
    for rf in r_factors:
        rf_dict = rf if isinstance(rf, dict) else rf.model_dump()
        print(f"    * [{rf_dict.get('category')}] {rf_dict.get('title')} (+{rf_dict.get('points')} points)")

    assert r_score == 10.0, f"Expected Risk Score 10.0, got {r_score}"
    assert r_level == "LOW", f"Expected Risk Level LOW, got {r_level}"
    assert len(r_factors) == 1, f"Expected exactly 1 risk factor, got {len(r_factors)}"

    # STEP F: AGENT 6 FINAL REPORT & DECISION
    state_input["risk_assessment_results"] = risk_data
    report_state = agent_6._graph.invoke(state_input)
    final_report = report_state.get("final_report")
    if hasattr(final_report, "model_dump"):
        fr_data = final_report.model_dump()
    elif isinstance(final_report, dict):
        fr_data = final_report
    else:
        fr_data = {}

    decision = fr_data.get("decision")
    reason = fr_data.get("decision_reason")
    doc_sum = fr_data.get("document_summary", {})
    val_sum = fr_data.get("validation_summary", {})

    print(f"\n[AGENT 6] Final Report & Decision Engine completed.")
    print(f"  - Final Decision: {decision}")
    print(f"  - Reason: {reason}")
    print(f"  - Upload Completeness: {doc_sum.get('satisfied_count')}/{doc_sum.get('required_count')} (Status: {doc_sum.get('status')})")
    print(f"  - Validation Errors: {val_sum.get('error_count')} (Blocking: {val_sum.get('blocking_count')})")

    assert decision == "HUMAN_REVIEW", f"Expected decision HUMAN_REVIEW, got {decision}"
    assert doc_sum.get("status") == "COMPLETE", f"Expected Upload Completeness COMPLETE, got {doc_sum.get('status')}"
    assert val_sum.get("error_count") == 1, f"Expected Validation Errors 1, got {val_sum.get('error_count')}"

    print("\n" + "=" * 80)
    print("✅ ALL 10 EDUCATION LOAN MULTI-AGENT PIPELINE TEST ASSERTIONS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    test_education_loan_pipeline_fixes()
