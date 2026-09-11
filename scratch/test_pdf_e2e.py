"""
End-to-end verification script for PDF report export.
"""

import sys
import time
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import app, ACTIVE_APPLICATIONS
from shared.state import (
    FinalReport,
    DocumentSummary,
    ValidationSummary,
    CrossDocumentSummary,
    RiskSummary
)

client = TestClient(app)

def test_pdf_e2e():
    app_id = f"APP-{int(time.time()*1000)}"
    report = FinalReport(
        application_id=app_id,
        loan_type="home_loan",
        decision="PASS",
        decision_reason="All mandatory documents satisfied and zero risk factors.",
        risk_score=0.0,
        risk_level="LOW",
        verification_coverage=100.0,
        consistency_score=100.0,
        document_summary=DocumentSummary(uploaded_count=3, required_count=3, satisfied_count=3, status="COMPLETE"),
        validation_summary=ValidationSummary(total_checks=15, passed_count=15),
        cross_document_summary=CrossDocumentSummary(verification_coverage=100.0, consistency_score=100.0, match_count=5),
        risk_summary=RiskSummary(risk_score=0.0, risk_level="LOW"),
        key_findings=["✅ Pristine application."],
        recommendations=["Approve loan."],
        executive_summary="Home Loan application processed cleanly.",
        generation_status="GENAI_GENERATED"
    )

    ACTIVE_APPLICATIONS[app_id] = {
        "application_id": app_id,
        "loan_type": "home_loan",
        "final_report_results": report.model_dump(),
        "status": "COMPLETED"
    }

    resp = client.get(f"/api/applications/{app_id}/report/pdf")
    assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
    assert resp.headers["content-type"] == "application/pdf"
    assert resp.content.startswith(b"%PDF-")
    print(f"PASSED End-to-End PDF Export Test for {app_id}! Generated PDF size: {len(resp.content)} bytes.")

if __name__ == "__main__":
    test_pdf_e2e()
