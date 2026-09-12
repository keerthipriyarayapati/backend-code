"""
Loan Policy Knowledge Base Package.
Provides centralized underwriting policies, rule definitions, and document requirements
with complete source provenance (DEMO_POLICY).
"""
from policy_kb.definitions import ALL_LOAN_POLICIES, get_policy_definition
from policy_kb.loader import seed_loan_policies, get_active_policy

__all__ = [
    "ALL_LOAN_POLICIES",
    "get_policy_definition",
    "seed_loan_policies",
    "get_active_policy"
]
