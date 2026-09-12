"""
Deterministic Rule Evaluation Logic for Loan Underwriting Policies.
Provides robust operator evaluation, currency parsing, and strict null handling.
"""

import re
import logging
from typing import Any, Tuple, Optional, List

logger = logging.getLogger("EligibilityRules")


def parse_numeric_value(val: Any) -> Optional[float]:
    """
    Parses a string or numeric value to float.
    Removes currency symbols (₹, $, INR, Rs), commas, percentage signs, and whitespace.
    Returns None if value cannot be parsed as a float.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ["none", "null", "n/a", "unknown", ""]:
        return None

    # Remove common currency and formatting symbols
    cleaned = re.sub(r"[₹\$,INRRs\s%]", "", val_str, flags=re.IGNORECASE).strip()
    # Handle patterns like "25,000.00" -> "25000.00"
    cleaned = cleaned.replace(",", "")
    
    try:
        return float(cleaned)
    except ValueError:
        # Check if first token is a number e.g. "25000 per month"
        m = re.search(r"[-+]?\d*\.?\d+", cleaned)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                pass
        return None


def is_missing_value(val: Any) -> bool:
    """Checks if a field value is missing or null, treating 0/0.0 as present."""
    if val is None:
        return True
    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in ["", "none", "null", "n/a", "unknown", "undefined"]:
            return True
    return False


def evaluate_rule_operator(
    operator: str,
    actual_val: Any,
    expected_val: Any,
    threshold_val: Optional[float] = None
) -> Tuple[str, Optional[str]]:
    """
    Evaluates a policy rule operator against an extracted actual value.
    Returns: (status, failure_reason)
    status: 'PASS', 'FAIL', 'INSUFFICIENT_EVIDENCE'
    """
    op = operator.strip().upper()

    # Rule: EXISTS
    if op == "EXISTS":
        if is_missing_value(actual_val):
            return "INSUFFICIENT_EVIDENCE", "Required evidence field is missing or not provided."
        return "PASS", None

    # For other operators, if actual_val is missing:
    if is_missing_value(actual_val):
        return "INSUFFICIENT_EVIDENCE", "Evidence value missing for policy evaluation."

    # Rule: GTE / LTE (Numeric comparison)
    if op in ["GTE", "LTE"]:
        actual_num = parse_numeric_value(actual_val)
        target_num = threshold_val if threshold_val is not None else parse_numeric_value(expected_val)

        if actual_num is None:
            return "INSUFFICIENT_EVIDENCE", f"Extracted value '{actual_val}' could not be parsed as a numeric quantity."
        if target_num is None:
            return "PASS", None

        if op == "GTE":
            if actual_num >= target_num:
                return "PASS", None
            else:
                return "FAIL", f"Value {actual_num} is below the policy minimum threshold of {target_num}."
        elif op == "LTE":
            if actual_num <= target_num:
                return "PASS", None
            else:
                return "FAIL", f"Value {actual_num} exceeds the policy maximum ceiling of {target_num}."

    # Rule: EQ (Equality)
    elif op == "EQ":
        actual_num = parse_numeric_value(actual_val)
        expected_num = parse_numeric_value(expected_val)
        if actual_num is not None and expected_num is not None:
            if abs(actual_num - expected_num) < 1e-4:
                return "PASS", None
            return "FAIL", f"Value {actual_num} does not match expected {expected_num}."
        else:
            if str(actual_val).strip().lower() == str(expected_val).strip().lower():
                return "PASS", None
            return "FAIL", f"Value '{actual_val}' does not equal expected '{expected_val}'."

    # Rule: IN (Membership)
    elif op == "IN":
        actual_str = str(actual_val).strip().lower()
        if isinstance(expected_val, list):
            options = [str(x).strip().lower() for x in expected_val]
        else:
            options = [x.strip().lower() for x in str(expected_val).split(",") if x.strip()]
        
        if actual_str in options:
            return "PASS", None
        return "FAIL", f"Value '{actual_val}' is not among authorized options: {options}."

    # Rule: MATCHES (Regex)
    elif op == "MATCHES":
        try:
            if re.search(str(expected_val), str(actual_val), re.IGNORECASE):
                return "PASS", None
            return "FAIL", f"Value '{actual_val}' does not conform to pattern '{expected_val}'."
        except Exception as e:
            logger.warning(f"Regex error in rule evaluation: {e}")
            return "INSUFFICIENT_EVIDENCE", f"Evaluation regex error: {e}"

    return "PASS", None
