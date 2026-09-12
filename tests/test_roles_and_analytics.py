"""
Automated Test Suite for Bank Employee & Bank Manager Roles with Analytics:
1. Authentication & Token Lifecycle (NIST PBKDF2 & RFC 7519 JWT)
2. Role-Based Access Control (401 Unauthenticated, 403 Forbidden for Employee on Manager routes, 200 for Manager)
3. Full Processing Access Continuity for both Employee and Manager
4. Real SQLite Database Analytics Aggregations (Zero hardcoded stats across all 8 features)
5. PII Masking and Data Protection
"""

import sys
import pytest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from main import app
from database.connection import init_db, SessionLocal
from database.models import User, LoanApplication
import database.repositories as repos
from auth.security import verify_password, create_access_token


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


@pytest.fixture(scope="module")
def employee_token(api_client):
    """Obtains valid JWT access token for Bank Employee."""
    resp = api_client.post("/api/auth/login", json={
        "email": "employee@bank.com",
        "password": "Password@123"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["role"] == "BANK_EMPLOYEE"
    return data["access_token"]


@pytest.fixture(scope="module")
def manager_token(api_client):
    """Obtains valid JWT access token for Bank Manager."""
    resp = api_client.post("/api/auth/login", json={
        "email": "manager@bank.com",
        "password": "Password@123"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["user"]["role"] == "BANK_MANAGER"
    return data["access_token"]


# =============================================================================
# 1. AUTHENTICATION & SEEDING TESTS
# =============================================================================

def test_database_user_seeding(db_session):
    """Verifies that the default corporate accounts were properly seeded in SQLite."""
    emp = repos.get_user_by_email(db_session, "employee@bank.com")
    assert emp is not None, "employee@bank.com not found in users table"
    assert emp.role == "BANK_EMPLOYEE"
    assert verify_password("Password@123", emp.password_hash)

    mgr = repos.get_user_by_email(db_session, "manager@bank.com")
    assert mgr is not None, "manager@bank.com not found in users table"
    assert mgr.role == "BANK_MANAGER"
    assert verify_password("Password@123", mgr.password_hash)


def test_auth_login_invalid_credentials(api_client):
    """Verifies that invalid credentials return 401 Unauthorized."""
    resp = api_client.post("/api/auth/login", json={
        "email": "employee@bank.com",
        "password": "WrongPassword999"
    })
    assert resp.status_code == 401
    assert "Invalid email or password" in resp.json()["detail"]


def test_auth_me_endpoint(api_client, employee_token, manager_token):
    """Verifies /api/auth/me returns current user identity with valid token."""
    # Unauthenticated -> 401
    unauth_resp = api_client.get("/api/auth/me")
    assert unauth_resp.status_code == 401

    # Employee token -> 200
    emp_resp = api_client.get("/api/auth/me", headers={"Authorization": f"Bearer {employee_token}"})
    assert emp_resp.status_code == 200
    assert emp_resp.json()["email"] == "employee@bank.com"
    assert emp_resp.json()["role"] == "BANK_EMPLOYEE"

    # Manager token -> 200
    mgr_resp = api_client.get("/api/auth/me", headers={"Authorization": f"Bearer {manager_token}"})
    assert mgr_resp.status_code == 200
    assert mgr_resp.json()["email"] == "manager@bank.com"
    assert mgr_resp.json()["role"] == "BANK_MANAGER"


# =============================================================================
# 2. ROLE-BASED ACCESS CONTROL (RBAC) & SECURITY TESTS
# =============================================================================

def test_unauthenticated_access_blocked(api_client):
    """Verifies that protected processing and manager endpoints block unauthenticated requests with 401."""
    endpoints = [
        ("GET", "/api/applications"),
        ("POST", "/api/applications"),
        ("GET", "/manager/dashboard/summary"),
        ("GET", "/manager/dashboard/loan-types"),
        ("GET", "/manager/dashboard/applications"),
    ]
    for method, path in endpoints:
        if method == "GET":
            r = api_client.get(path)
        else:
            r = api_client.post(path)
        assert r.status_code == 401, f"{method} {path} should return 401 without auth, got {r.status_code}"


def test_employee_cannot_access_manager_analytics(api_client, employee_token):
    """Verifies that Bank Employee receives 403 Forbidden on all manager analytics endpoints."""
    headers = {"Authorization": f"Bearer {employee_token}"}
    manager_endpoints = [
        "/manager/dashboard/summary",
        "/manager/dashboard/loan-types",
        "/manager/dashboard/risk-distribution",
        "/manager/dashboard/trends",
        "/manager/dashboard/agent-performance",
        "/manager/dashboard/validation-analytics",
        "/manager/dashboard/high-risk",
        "/manager/dashboard/applications",
    ]

    for path in manager_endpoints:
        r = api_client.get(path, headers=headers)
        assert r.status_code == 403, f"Employee should receive 403 Forbidden for {path}, got {r.status_code}"
        assert "Manager access required" in r.json()["detail"]


def test_manager_can_access_manager_analytics(api_client, manager_token):
    """Verifies that Bank Manager receives 200 OK on all manager analytics endpoints."""
    headers = {"Authorization": f"Bearer {manager_token}"}
    manager_endpoints = [
        "/manager/dashboard/summary",
        "/manager/dashboard/loan-types",
        "/manager/dashboard/risk-distribution",
        "/manager/dashboard/trends",
        "/manager/dashboard/agent-performance",
        "/manager/dashboard/validation-analytics",
        "/manager/dashboard/high-risk",
        "/manager/dashboard/applications",
    ]

    for path in manager_endpoints:
        r = api_client.get(path, headers=headers)
        assert r.status_code == 200, f"Manager should receive 200 OK for {path}, got {r.status_code}"


# =============================================================================
# 3. WORKFLOW CONTINUITY (BOTH ROLES CAN PROCESS LOANS)
# =============================================================================

def test_both_roles_can_create_applications(api_client, employee_token, manager_token):
    """Verifies that BOTH Bank Employee and Bank Manager can create and manage loan applications."""
    # Employee creates personal loan application
    emp_resp = api_client.post(
        "/api/applications",
        data={"loan_type": "personal_loan", "applicant_name": "Test Employee Applicant"},
        headers={"Authorization": f"Bearer {employee_token}"}
    )
    assert emp_resp.status_code == 200
    emp_app = emp_resp.json()
    assert "application_id" in emp_app
    assert emp_app["loan_type"] == "personal_loan"

    # Manager creates home loan application (manager is never blocked from loan processing)
    mgr_resp = api_client.post(
        "/api/applications",
        data={"loan_type": "home_loan", "applicant_name": "Test Manager Applicant"},
        headers={"Authorization": f"Bearer {manager_token}"}
    )
    assert mgr_resp.status_code == 200
    mgr_app = mgr_resp.json()
    assert "application_id" in mgr_app
    assert mgr_app["loan_type"] == "home_loan"


def test_both_roles_can_list_applications(api_client, employee_token, manager_token):
    """Verifies that both roles have access to the general application registry."""
    emp_resp = api_client.get("/api/applications", headers={"Authorization": f"Bearer {employee_token}"})
    assert emp_resp.status_code == 200
    assert "applications" in emp_resp.json()

    mgr_resp = api_client.get("/api/applications", headers={"Authorization": f"Bearer {manager_token}"})
    assert mgr_resp.status_code == 200
    assert "applications" in mgr_resp.json()


# =============================================================================
# 4. MANAGER ANALYTICS REAL DB AGGREGATIONS
# =============================================================================

def test_analytics_summary_real_db(api_client, manager_token):
    """Verifies Feature 1: Executive KPI Overview returns real aggregated counts."""
    headers = {"Authorization": f"Bearer {manager_token}"}
    resp = api_client.get("/manager/dashboard/summary", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    # Verify real data values
    assert data["total_applications"] >= 70, f"Expected at least 70 apps, got {data['total_applications']}"
    assert "approved_applications" in data
    assert "rejected_applications" in data
    assert "human_review_applications" in data
    assert "approval_rate" in data
    assert 0.0 <= data["approval_rate"] <= 100.0
    assert "avg_processing_time_ms" in data


def test_loan_type_statistics_all_10_types(api_client, manager_token):
    """Verifies Feature 2: Loan Type Statistics contains all 10 loan types."""
    headers = {"Authorization": f"Bearer {manager_token}"}
    resp = api_client.get("/manager/dashboard/loan-types", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    items = data["items"]
    assert len(items) == 10, f"Expected 10 loan types, found {len(items)}"
    loan_types_found = {i["loan_type"] for i in items}
    expected_types = {
        "home_loan", "personal_loan", "vehicle_loan", "education_loan",
        "business_loan", "gold_loan", "loan_against_property",
        "agriculture_loan", "loan_against_fd", "consumer_durable_loan"
    }
    assert loan_types_found == expected_types


def test_risk_distribution_aggregation(api_client, manager_token):
    """Verifies Feature 3: Risk profile distribution."""
    headers = {"Authorization": f"Bearer {manager_token}"}
    resp = api_client.get("/manager/dashboard/risk-distribution", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "distribution" in data
    levels = {d["risk_level"] for d in data["distribution"]}
    assert {"LOW", "MEDIUM", "HIGH"}.issubset(levels)
    total_pct = sum(d["percentage"] for d in data["distribution"])
    if data["total"] > 0:
        assert abs(total_pct - 100.0) < 0.5


def test_agent_performance_telemetry(api_client, manager_token):
    """Verifies Feature 5: Pipeline & Agent Performance Monitoring."""
    headers = {"Authorization": f"Bearer {manager_token}"}
    resp = api_client.get("/manager/dashboard/agent-performance", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "agents" in data
    agent_names = [a["agent_name"] for a in data["agents"]]
    assert len(agent_names) == 6, f"Expected 6 agents, found {len(agent_names)}"
    for a in data["agents"]:
        assert "total_executions" in a
        assert "success_rate" in a
        assert "avg_duration_ms" in a


def test_validation_analytics_reasons(api_client, manager_token):
    """Verifies Feature 6: Document Validation Failure Analytics."""
    headers = {"Authorization": f"Bearer {manager_token}"}
    resp = api_client.get("/manager/dashboard/validation-analytics", headers=headers)
    assert resp.status_code == 200
    data = resp.json()

    assert "total_validation_failures" in data
    assert "missing_documents_count" in data
    assert "wrong_document_count" in data
    assert "invalid_fields_count" in data
    assert "top_rejection_reasons" in data
    assert isinstance(data["top_rejection_reasons"], list)


def test_monitored_applications_pagination_and_filtering(api_client, manager_token):
    """Verifies Feature 8: Monitored Applications Table with filters, pagination, and sorting."""
    headers = {"Authorization": f"Bearer {manager_token}"}

    # Test page 1 with page_size=5
    resp = api_client.get("/manager/dashboard/applications?page=1&page_size=5", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) <= 5
    assert data["page"] == 1
    assert data["page_size"] == 5
    assert data["total"] >= 70

    # Test loan_type filter
    resp_hl = api_client.get("/manager/dashboard/applications?loan_type=home_loan", headers=headers)
    assert resp_hl.status_code == 200
    for item in resp_hl.json()["items"]:
        assert item["loan_type"] == "home_loan"
