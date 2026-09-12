"""
Comprehensive Automated Test Suite for Advanced Explainable Loan Processing Features:
1. Loan Eligibility Engine (Deterministic rules & 9-tier priority)
2. Loan Policy Knowledge Base (10 Loan Types, DEMO_POLICY citations)
3. Field-Level & Decision Evidence Citations (Snippets, Page Numbers, PII Masking)
4. Evidence-Based Decision Graph (11 Stages, 10 Directed Edges, Node Statuses)
5. Telemetry & Performance Metrics (True measured timings, Memory, Agent logs)
6. REST API Endpoints & PDF Exporter Integration
"""

import sys
import time
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import app
from database.connection import init_db, SessionLocal
from database.models import LoanApplication, LoanPolicyModel, PolicyRuleModel
import database.repositories as repos

from policy_kb.definitions import ALL_LOAN_POLICIES, get_policy_definition
from policy_kb.loader import get_active_policy, seed_loan_policies
from eligibility.rules import evaluate_rule_operator, parse_numeric_value, is_missing_value
from eligibility.engine import LoanEligibilityEngine, evaluate_application_eligibility
from evidence.manager import create_field_evidence, build_rule_citations, mask_pii_value
from decision_graph.builder import DecisionGraphBuilder, build_application_decision_graph
from telemetry.tracker import PipelineTelemetryTracker, get_current_memory_mb


@pytest.fixture(scope="module")
def db_session():
    """Initializes the database and provides a scoped session."""
    init_db()
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def api_client():
    """FastAPI TestClient instance."""
    return TestClient(app)


# =============================================================================
# 1. POLICY KNOWLEDGE BASE TESTS
# =============================================================================

def test_policy_knowledge_base_completeness(db_session):
    """Verifies that all 10 loan types have policies with DEMO_POLICY citations."""
    policies = repos.list_all_policies(db_session)
    assert len(policies) == 10, f"Expected 10 policies in DB, found {len(policies)}"

    for pol in policies:
        assert pol["source_type"] == "DEMO_POLICY", f"Policy {pol['policy_id']} missing DEMO_POLICY source_type"
        assert pol["source_document"] is not None
        assert pol["source_section"] is not None
        assert pol["source_page"] is not None
        assert len(pol["rules"]) >= 3, f"Policy {pol['policy_id']} has fewer than 3 rules"

        for r in pol["rules"]:
            assert r["source_type"] == "DEMO_POLICY"
            assert r["source_document"] is not None
            assert r["source_section"] is not None
            assert r["source_page"] is not None
            assert r["operator"] in ["GTE", "LTE", "EQ", "IN", "EXISTS", "MATCHES"]


def test_policy_query_by_loan_type(db_session):
    """Verifies querying policies for specific loan types."""
    test_types = [
        "Home Loan",
        "Personal Loan",
        "Car / Vehicle Loan",
        "Education Loan",
        "Business Loan",
        "Gold Loan",
        "Loan Against Property",
        "Agriculture / Crop Loan",
        "Loan Against Fixed Deposit",
        "Consumer Durable Loan"
    ]
    for lt in test_types:
        pol = get_active_policy(db_session, lt)
        assert pol is not None, f"Failed to retrieve policy for '{lt}'"
        assert pol["loan_type"].lower() == lt.lower()


# =============================================================================
# 2. FIELD EVIDENCE & PII MASKING TESTS
# =============================================================================

def test_pii_masking():
    """Verifies sensitive values are masked properly (only last 4 characters visible)."""
    # PAN
    assert mask_pii_value("ABCDE1234F", "pan_number") == "XXXXXX234F"
    # Account Number
    masked_acc = mask_pii_value("123456789012", "account_number")
    assert masked_acc.endswith("9012")
    assert "123456" not in masked_acc
    # Aadhaar
    masked_aadh = mask_pii_value("123456789012", "aadhaar_number")
    assert masked_aadh.endswith("9012")
    # Non-sensitive
    assert mask_pii_value("John Doe", "applicant_name") == "John Doe"


def test_field_evidence_creation():
    """Verifies field evidence citations record provenance and mask sensitive snippets."""
    ev = create_field_evidence(
        application_id="APP_TEST_001",
        field_name="pan_number",
        raw_value="ABCDE1234F",
        document_id=1,
        slot_id="pan_slot",
        document_type="pan_card",
        source_page=1,
        snippet="Permanent Account Number: ABCDE1234F recorded on card.",
        extraction_method="Regex / Key-Value Extractor",
        ocr_used=False,
        confidence=0.98
    )
    assert ev.evidence_id.startswith("EV_APP_TEST_001_pan_number_")
    assert ev.source_page == 1
    assert ev.confidence == 0.98
    assert "ABCDE1234F" not in ev.extracted_value  # Masked!
    assert "ABCDE1234F" not in ev.snippet          # Snippet also masked!


def test_decision_citations_builder():
    """Verifies decision citations link policy rules and field evidence."""
    ev = create_field_evidence(
        application_id="APP_TEST_001",
        field_name="monthly_income",
        raw_value="50000",
        document_type="payslip",
        source_page=1,
        snippet="Net Pay: Rs. 50,000.00"
    )
    rule_def = {
        "rule_id": "HL_R01_MIN_INCOME",
        "rule_code": "HL_MIN_INCOME",
        "source_document": "Retail Credit Manual",
        "source_section": "Section 4.1",
        "source_page": 14,
        "operator": "GTE",
        "expected_value": "25000"
    }
    citations = build_rule_citations("HL_MIN_INCOME", rule_def, ev)
    assert len(citations) == 2
    assert citations[0].citation_type == "POLICY_RULE"
    assert citations[1].citation_type == "FIELD"
    assert "Retail Credit Manual" in citations[0].source_title


# =============================================================================
# 3. ELIGIBILITY ENGINE DETERMINISTIC RULES & 9-TIER PRIORITY TESTS
# =============================================================================

def test_numeric_parsing():
    """Verifies robust parsing of currency, commas, and percentage strings."""
    assert parse_numeric_value("₹25,000.00") == 25000.0
    assert parse_numeric_value("$1,500") == 1500.0
    assert parse_numeric_value("INR 35000") == 35000.0
    assert parse_numeric_value("65%") == 65.0
    assert parse_numeric_value("none") is None
    assert parse_numeric_value(None) is None
    assert parse_numeric_value(0) == 0.0


def test_rule_operator_evaluation():
    """Verifies operator evaluations for GTE, LTE, EQ, IN, EXISTS."""
    assert evaluate_rule_operator("GTE", "30000", "25000")[0] == "PASS"
    assert evaluate_rule_operator("GTE", "20000", "25000")[0] == "FAIL"
    assert evaluate_rule_operator("LTE", "55", "60")[0] == "PASS"
    assert evaluate_rule_operator("LTE", "65", "60")[0] == "FAIL"
    assert evaluate_rule_operator("EQ", "SALARIED", "salaried")[0] == "PASS"
    assert evaluate_rule_operator("IN", "Tier-1", "Tier-1, Tier-2")[0] == "PASS"
    assert evaluate_rule_operator("EXISTS", "Valid Title", None)[0] == "PASS"
    assert evaluate_rule_operator("EXISTS", None, None)[0] == "INSUFFICIENT_EVIDENCE"


def test_eligibility_hierarchy_priority(db_session):
    """
    Verifies the strict 9-point decision hierarchy in exact order:
    1. Missing doc -> INSUFFICIENT_EVIDENCE
    2. Wrong doc -> INSUFFICIENT_EVIDENCE
    3. Critical validation failure -> HUMAN_REVIEW_REQUIRED
    4. Critical cross-doc mismatch -> HUMAN_REVIEW_REQUIRED
    5. High risk score -> HUMAN_REVIEW_REQUIRED
    6. Policy rule failed -> NOT_ELIGIBLE
    7. Clean pass -> ELIGIBLE
    """
    engine = LoanEligibilityEngine(db=db_session)

    # 1. Missing Required Document -> INSUFFICIENT_EVIDENCE
    res1 = engine.evaluate(
        application_id="APP_HIER_01",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "missing"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={"monthly_income": 50000}
    )
    assert res1.decision == "INSUFFICIENT_EVIDENCE"

    # 2. Wrong Required Document -> INSUFFICIENT_EVIDENCE
    res2 = engine.evaluate(
        application_id="APP_HIER_02",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "wrong_document"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={"monthly_income": 50000}
    )
    assert res2.decision == "INSUFFICIENT_EVIDENCE"

    # 3. Critical Validation Failure -> HUMAN_REVIEW_REQUIRED
    res3 = engine.evaluate(
        application_id="APP_HIER_03",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "accepted"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={"monthly_income": 50000, "pan_number": "ABCDE1234F"},
        validation_results=[
            {"status": "FAIL", "severity": "CRITICAL", "message": "PAN format checksum corrupted"}
        ]
    )
    assert res3.decision == "HUMAN_REVIEW_REQUIRED"

    # 4. Critical Cross-Document Mismatch -> HUMAN_REVIEW_REQUIRED
    res4 = engine.evaluate(
        application_id="APP_HIER_04",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "accepted"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={"monthly_income": 50000, "pan_number": "ABCDE1234F"},
        cross_document_findings=[
            {"match_status": "MISMATCH", "severity": "CRITICAL", "discrepancy_details": "PAN mismatch between KYC and Payslip"}
        ]
    )
    assert res4.decision == "HUMAN_REVIEW_REQUIRED"

    # 5. Elevated Risk (Score >= 50) -> HUMAN_REVIEW_REQUIRED
    res5 = engine.evaluate(
        application_id="APP_HIER_05",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "accepted"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={"monthly_income": 50000, "pan_number": "ABCDE1234F"},
        risk_assessment={"risk_score": 65.0, "risk_level": "HIGH"}
    )
    assert res5.decision == "HUMAN_REVIEW_REQUIRED"

    # 6. Policy Rule Breach -> NOT_ELIGIBLE
    # Home loan requires monthly_income >= 25000. Provide 15000.
    res6 = engine.evaluate(
        application_id="APP_HIER_06",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "accepted"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={"monthly_income": 15000, "applicant_age": 30, "cibil_score": 750, "pan_number": "ABCDE1234F"}
    )
    assert res6.decision == "NOT_ELIGIBLE"
    assert any("HL_MIN_INCOME" in r for r in res6.reasons)

    # 7. Clean Pass -> ELIGIBLE
    res7 = engine.evaluate(
        application_id="APP_HIER_07",
        loan_type="Home Loan",
        document_slots=[
            {"display_name": "KYC", "required": True, "status": "accepted"},
            {"display_name": "Income Proof", "required": True, "status": "accepted"}
        ],
        extracted_fields={
            "monthly_income": 50000,
            "applicant_age": 32,
            "cibil_score": 780,
            "pan_number": "ABCDE1234F",
            "name": "John Doe",
            "net_salary": 50000,
            "account_number": "1234567890",
            "property_address": "123 Green Valley, Bangalore"
        }
    )
    assert res7.decision == "ELIGIBLE"


# =============================================================================
# 4. DECISION GRAPH TESTS
# =============================================================================

def test_decision_graph_topology(db_session):
    """Verifies the 11-stage decision graph topology and edge sequencing."""
    builder = DecisionGraphBuilder(db=db_session)
    graph = builder.build(
        application_id="APP_GRAPH_01",
        loan_type="Personal Loan",
        document_slots=[{"display_name": "KYC", "required": True, "status": "accepted"}],
        classifications=[{"document_type": "kyc_identity", "confidence": 0.95}],
        extracted_fields={"monthly_income": 35000},
        validation_results=[{"status": "PASS", "severity": "LOW"}],
        cross_document_findings=[],
        risk_assessment={"risk_score": 10.0, "risk_level": "LOW"}
    )

    assert graph.total_nodes == 11
    assert len(graph.nodes) == 11
    assert graph.total_edges == 10
    assert len(graph.edges) == 10

    expected_node_ids = [
        "app_received",
        "loan_type_identified",
        "required_docs_check",
        "agent_1_classify",
        "agent_2_extract",
        "agent_3_validate",
        "agent_4_cross_doc",
        "agent_5_risk",
        "policy_evaluation",
        "eligibility_engine",
        "agent_6_final_decision"
    ]
    actual_node_ids = [n.node_id for n in graph.nodes]
    assert actual_node_ids == expected_node_ids

    # Edge path verification
    for i in range(len(expected_node_ids) - 1):
        assert graph.edges[i].source == expected_node_ids[i]
        assert graph.edges[i].target == expected_node_ids[i + 1]


# =============================================================================
# 5. TELEMETRY TRACKER TESTS
# =============================================================================

def test_telemetry_tracker_measurements(db_session):
    """Verifies that telemetry measures real elapsed time using time.perf_counter()."""
    tracker = PipelineTelemetryTracker("APP_TELEM_01", db=db_session)

    with tracker.track_agent("agent_1", {"input": "test"}):
        time.sleep(0.01)  # 10ms real work

    with tracker.track_agent("agent_2"):
        time.sleep(0.01)

    telemetry = tracker.finalize(documents_count=2, fields_count=8, citations_count=4)

    assert telemetry.total_pipeline_duration_ms > 15.0  # Real measured time
    assert "agent_1" in telemetry.agent_durations
    assert "agent_2" in telemetry.agent_durations
    assert telemetry.agent_durations["agent_1"] >= 9.0
    assert telemetry.agent_durations["agent_2"] >= 9.0
    assert len(telemetry.agent_logs) == 2
    assert telemetry.memory_mb >= 0.0


# =============================================================================
# 6. REST API ENDPOINTS & PDF EXPORTER INTEGRATION
# =============================================================================

def test_rest_api_policies(api_client):
    """Verifies /api/policies and /api/policies/{loan_type} endpoints."""
    # List all
    res = api_client.get("/api/policies")
    assert res.status_code == 200
    data = res.json()
    assert "policies" in data
    assert len(data["policies"]) == 10

    # Get specific loan type
    res_hl = api_client.get("/api/policies/home_loan")
    assert res_hl.status_code == 200
    assert res_hl.json()["loan_type"] == "Home Loan"
    assert res_hl.json()["source_type"] == "DEMO_POLICY"

    res_cd = api_client.get("/api/policies/consumer_durable_loan")
    assert res_cd.status_code == 200
    assert res_cd.json()["loan_type"] == "Consumer Durable Loan"


def test_rest_api_application_artifacts(api_client, db_session):
    """Verifies endpoints for eligibility, evidence, decision-graph, telemetry, and final report."""
    app_id = f"APP_API_TEST_{int(time.time())}"
    repos.create_application(db_session, application_id=app_id, loan_type="Personal Loan")

    # Seed an eligibility result
    repos.save_eligibility_result(
        db_session,
        eligibility_data={
            "eligibility_id": f"EL_{app_id}",
            "application_id": app_id,
            "loan_type": "Personal Loan",
            "decision": "ELIGIBLE",
            "confidence": 0.95,
            "reasons": ["Test passed"],
            "rules_evaluated_count": 4,
            "rules_passed_count": 4,
            "rules_failed_count": 0,
            "rules_skipped_count": 0,
            "processing_time_ms": 12.5
        },
        rule_results_data=[
            {
                "rule_code": "PL_MIN_INCOME",
                "category": "INCOME",
                "operator": "GTE",
                "expected_value": "20000",
                "actual_value": "35000",
                "status": "PASS",
                "mandatory": True,
                "severity": "CRITICAL"
            }
        ]
    )

    # Seed field evidence
    repos.save_field_evidence_batch(
        db_session,
        [
            {
                "evidence_id": f"EV_{app_id}_pan",
                "application_id": app_id,
                "field_name": "pan_number",
                "extracted_value": "XXXXXX1234F",
                "source_page": 1,
                "snippet": "PAN Number: XXXXXX1234F",
                "confidence": 0.98
            }
        ]
    )

    # Seed decision graph
    builder = DecisionGraphBuilder(db=db_session)
    builder.build(
        application_id=app_id,
        loan_type="Personal Loan",
        document_slots=[],
        classifications=[],
        extracted_fields={},
        validation_results=[],
        cross_document_findings=[],
        risk_assessment={"risk_score": 15.0}
    )

    # Seed telemetry
    repos.save_telemetry_metrics(
        db_session,
        application_id=app_id,
        metrics={
            "total_pipeline_duration_ms": 124.5,
            "agent_durations": {"agent_1": 20.0, "agent_2": 35.0},
            "memory_mb": 45.0,
            "documents_count": 2,
            "fields_count": 10,
            "citations_count": 4
        }
    )

    # Test GET /api/applications/{app_id}/eligibility
    res_el = api_client.get(f"/api/applications/{app_id}/eligibility")
    assert res_el.status_code == 200
    assert res_el.json()["decision"] == "ELIGIBLE"
    assert len(res_el.json()["rule_results"]) == 1

    # Test GET /api/applications/{app_id}/evidence
    res_ev = api_client.get(f"/api/applications/{app_id}/evidence")
    assert res_ev.status_code == 200
    assert res_ev.json()["total_citations"] >= 1

    # Test GET /api/applications/{app_id}/decision-graph
    res_dg = api_client.get(f"/api/applications/{app_id}/decision-graph")
    assert res_dg.status_code == 200
    assert res_dg.json()["total_nodes"] == 11

    # Test GET /api/applications/{app_id}/telemetry
    res_telem = api_client.get(f"/api/applications/{app_id}/telemetry")
    assert res_telem.status_code == 200
    assert res_telem.json()["total_pipeline_duration_ms"] == 124.5
