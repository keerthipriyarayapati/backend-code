"""
Loan Eligibility Engine Package.
Provides deterministic credit underwriting rule evaluation, evidence citation linking,
and hierarchy-based decision outcomes.
"""

from eligibility.schemas import RuleEvaluationResult, EligibilityDecision
from eligibility.rules import evaluate_rule_operator, parse_numeric_value
from eligibility.engine import LoanEligibilityEngine, evaluate_application_eligibility

__all__ = [
    "RuleEvaluationResult",
    "EligibilityDecision",
    "evaluate_rule_operator",
    "parse_numeric_value",
    "LoanEligibilityEngine",
    "evaluate_application_eligibility"
]
