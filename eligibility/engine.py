"""
Deterministic Loan Eligibility Engine.
Evaluates loan applications against active underwriting policies, required documents,
field evidence, validation findings, cross-document discrepancies, and risk assessments.
Strictly implements the 9-tier decision priority hierarchy.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from eligibility.schemas import RuleEvaluationResult, EligibilityDecision
from eligibility.rules import evaluate_rule_operator, is_missing_value
from policy_kb.loader import get_active_policy
from database.repositories import (
    save_eligibility_result,
    get_eligibility_by_application,
    get_field_evidence_by_application
)

logger = logging.getLogger("LoanEligibilityEngine")


class LoanEligibilityEngine:
    """Deterministic, policy-driven loan eligibility engine."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def evaluate(
        self,
        application_id: str,
        loan_type: str,
        document_slots: List[Dict[str, Any]],
        extracted_fields: Dict[str, Any],
        validation_results: Optional[List[Dict[str, Any]]] = None,
        cross_document_findings: Optional[List[Dict[str, Any]]] = None,
        risk_assessment: Optional[Dict[str, Any]] = None,
        field_evidences: Optional[List[Dict[str, Any]]] = None
    ) -> EligibilityDecision:
        """
        Executes full eligibility evaluation according to the 9-point decision hierarchy:
        1. Missing required document -> INSUFFICIENT_EVIDENCE
        2. Wrong required document -> INSUFFICIENT_EVIDENCE
        3. Required field missing -> INSUFFICIENT_EVIDENCE
        4. Critical validation failure -> HUMAN_REVIEW_REQUIRED
        5. High-severity cross-document mismatch -> HUMAN_REVIEW_REQUIRED
        6. High or critical risk -> HUMAN_REVIEW_REQUIRED
        7. Policy rule clearly failed -> NOT_ELIGIBLE
        8. All required evidence available and all rules pass -> ELIGIBLE
        9. Evidence insufficient to evaluate -> INSUFFICIENT_EVIDENCE
        """
        start_time = time.perf_counter()
        eligibility_id = f"EL_{application_id}_{uuid.uuid4().hex[:6]}"
        validation_results = validation_results or []
        cross_document_findings = cross_document_findings or []
        risk_assessment = risk_assessment or {}
        
        # 1. Fetch policy for this loan type
        policy = None
        if self.db:
            policy = get_active_policy(self.db, loan_type)
        if not policy:
            from policy_kb.definitions import get_policy_definition
            policy = get_policy_definition(loan_type)

        reasons: List[str] = []
        rule_results: List[RuleEvaluationResult] = []
        decision = "ELIGIBLE"
        confidence = 1.0

        # Build evidence lookup by field_name
        ev_lookup: Dict[str, Dict[str, Any]] = {}
        if field_evidences:
            for ev in field_evidences:
                fn = ev.get("field_name")
                if fn and fn not in ev_lookup:
                    ev_lookup[fn] = ev

        # Flatten extracted fields for evaluation
        flat_fields: Dict[str, Any] = {}
        for k, v in extracted_fields.items():
            if isinstance(v, dict):
                flat_fields[k] = v.get("value")
            else:
                flat_fields[k] = v

        # -------------------------------------------------------------
        # STAGE 1 & 2: REQUIRED DOCUMENT CHECKS (Missing / Wrong Document)
        # -------------------------------------------------------------
        missing_required_docs = []
        wrong_required_docs = []

        for slot in document_slots:
            is_req = slot.get("required", True)
            status = str(slot.get("status", "")).lower()
            slot_name = slot.get("display_name") or slot.get("requirement_id")

            if is_req:
                if status in ["pending", "missing"]:
                    missing_required_docs.append(slot_name)
                elif status in ["wrong_document", "rejected"]:
                    wrong_required_docs.append(slot_name)

        if missing_required_docs:
            decision = "INSUFFICIENT_EVIDENCE"
            reasons.append(f"Missing mandatory required document(s): {', '.join(missing_required_docs)}")

        if wrong_required_docs and decision == "ELIGIBLE":
            decision = "INSUFFICIENT_EVIDENCE"
            reasons.append(f"Incorrect document submitted for required slot(s): {', '.join(wrong_required_docs)}")

        # -------------------------------------------------------------
        # STAGE 3: REQUIRED FIELDS CHECK
        # -------------------------------------------------------------
        required_fields_defs = policy.get("required_fields", []) if policy else []
        missing_fields = []
        for rf in required_fields_defs:
            if rf.get("required", True):
                fname = rf.get("field_name")
                if fname and (fname not in flat_fields or is_missing_value(flat_fields.get(fname))):
                    missing_fields.append(fname)

        if missing_fields and decision == "ELIGIBLE":
            decision = "INSUFFICIENT_EVIDENCE"
            reasons.append(f"Required information missing from documents: {', '.join(missing_fields)}")

        # -------------------------------------------------------------
        # STAGE 4: CRITICAL VALIDATION CHECK (Agent 3)
        # -------------------------------------------------------------
        critical_val_failures = []
        for vr in validation_results:
            status = str(vr.get("status", "")).upper()
            severity = str(vr.get("severity", "")).upper()
            if status == "FAIL" and severity in ["CRITICAL", "HIGH"]:
                critical_val_failures.append(vr.get("message") or vr.get("rule_name") or "Validation error")

        if critical_val_failures and decision == "ELIGIBLE":
            decision = "HUMAN_REVIEW_REQUIRED"
            reasons.append(f"Critical document validation alerts: {'; '.join(critical_val_failures[:3])}")

        # -------------------------------------------------------------
        # STAGE 5: CROSS-DOCUMENT MISMATCH CHECK (Agent 4)
        # -------------------------------------------------------------
        critical_cross_mismatches = []
        for cf in cross_document_findings:
            match_status = str(cf.get("match_status") or cf.get("status") or "").upper()
            severity = str(cf.get("severity", "")).upper()
            if match_status == "MISMATCH" and severity in ["CRITICAL", "HIGH"]:
                critical_cross_mismatches.append(
                    cf.get("discrepancy_details") or cf.get("message") or f"Mismatch on {cf.get('field_name')}"
                )

        if critical_cross_mismatches and decision in ["ELIGIBLE"]:
            decision = "HUMAN_REVIEW_REQUIRED"
            reasons.append(f"High-severity cross-document conflicts: {'; '.join(critical_cross_mismatches[:3])}")

        # -------------------------------------------------------------
        # STAGE 6: RISK ASSESSMENT CHECK (Agent 5)
        # -------------------------------------------------------------
        risk_score = float(risk_assessment.get("risk_score", 0.0) or 0.0)
        risk_level = str(risk_assessment.get("risk_level", "LOW")).upper()
        if (risk_score >= 50.0 or risk_level in ["HIGH", "CRITICAL"]) and decision == "ELIGIBLE":
            decision = "HUMAN_REVIEW_REQUIRED"
            reasons.append(f"Elevated credit risk assessment (Risk Score: {risk_score:.1f}, Level: {risk_level})")

        # -------------------------------------------------------------
        # STAGE 7: EVALUATE POLICY RULES
        # -------------------------------------------------------------
        policy_rules = policy.get("rules", []) if policy else []
        rules_evaluated = len(policy_rules)
        rules_passed = 0
        rules_failed = 0
        rules_skipped = 0

        for r_def in policy_rules:
            rcode = r_def.get("rule_code", "RULE")
            cat = r_def.get("category", "GENERAL")
            fname = r_def.get("field_name")
            op = r_def.get("operator", "EQ")
            exp_val = r_def.get("expected_value")
            thresh_val = r_def.get("threshold_value")
            sev = r_def.get("severity", "CRITICAL")
            mand = r_def.get("mandatory", True)
            rule_id = r_def.get("rule_id")

            actual_val = flat_fields.get(fname) if fname else None
            ev_record = ev_lookup.get(fname) if fname else None
            ev_id = ev_record.get("evidence_id") if ev_record else None

            # Evaluate operator
            st, fail_msg = evaluate_rule_operator(
                operator=op,
                actual_val=actual_val,
                expected_val=exp_val,
                threshold_val=thresh_val
            )

            if st == "PASS":
                rules_passed += 1
            elif st == "FAIL":
                rules_failed += 1
                if mand and sev in ["CRITICAL", "HIGH"]:
                    # Policy rule failure overrides ELIGIBLE and HUMAN_REVIEW to NOT_ELIGIBLE!
                    if decision != "INSUFFICIENT_EVIDENCE":
                        decision = "NOT_ELIGIBLE"
                    reasons.append(f"Policy Rule '{rcode}' Failed: {fail_msg}")
            elif st == "INSUFFICIENT_EVIDENCE":
                rules_skipped += 1
                if mand and decision == "ELIGIBLE":
                    decision = "INSUFFICIENT_EVIDENCE"
                    reasons.append(f"Insufficient evidence for mandatory rule '{rcode}' ({fname or 'field missing'}).")

            rule_results.append(
                RuleEvaluationResult(
                    rule_code=rcode,
                    category=cat,
                    field_name=fname,
                    operator=op,
                    expected_value=str(exp_val) if exp_val is not None else None,
                    actual_value=str(actual_val) if actual_val is not None else None,
                    status=st,
                    mandatory=mand,
                    severity=sev,
                    failure_reason=fail_msg,
                    evidence_id=ev_id,
                    policy_rule_id=rule_id
                )
            )

        # -------------------------------------------------------------
        # STAGE 8 & 9: DEFAULT CONCLUSION
        # -------------------------------------------------------------
        if decision == "ELIGIBLE":
            if not reasons:
                reasons.append("All mandatory underwriting policy criteria and document requirements verified successfully.")
            confidence = 0.95
        elif decision == "NOT_ELIGIBLE":
            confidence = 0.90
        elif decision == "HUMAN_REVIEW_REQUIRED":
            confidence = 0.85
        elif decision == "INSUFFICIENT_EVIDENCE":
            confidence = 0.75

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        outcome = EligibilityDecision(
            eligibility_id=eligibility_id,
            application_id=application_id,
            loan_type=loan_type,
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            rules_evaluated_count=rules_evaluated,
            rules_passed_count=rules_passed,
            rules_failed_count=rules_failed,
            rules_skipped_count=rules_skipped,
            processing_time_ms=duration_ms,
            rule_results=rule_results
        )

        # Persist to database if session provided
        if self.db:
            try:
                el_data = {
                    "eligibility_id": outcome.eligibility_id,
                    "application_id": outcome.application_id,
                    "loan_type": outcome.loan_type,
                    "decision": outcome.decision,
                    "confidence": outcome.confidence,
                    "reasons": outcome.reasons,
                    "rules_evaluated_count": outcome.rules_evaluated_count,
                    "rules_passed_count": outcome.rules_passed_count,
                    "rules_failed_count": outcome.rules_failed_count,
                    "rules_skipped_count": outcome.rules_skipped_count,
                    "processing_time_ms": outcome.processing_time_ms
                }
                rr_data = [rr.dict() for rr in outcome.rule_results]
                save_eligibility_result(self.db, el_data, rr_data)
                logger.info(f"Persisted eligibility evaluation {eligibility_id}: {decision}")
            except Exception as e:
                logger.error(f"Failed to persist eligibility decision: {e}", exc_info=True)

        return outcome


def evaluate_application_eligibility(
    db: Optional[Session],
    application_id: str,
    loan_type: str,
    document_slots: List[Dict[str, Any]],
    extracted_fields: Dict[str, Any],
    validation_results: Optional[List[Dict[str, Any]]] = None,
    cross_document_findings: Optional[List[Dict[str, Any]]] = None,
    risk_assessment: Optional[Dict[str, Any]] = None,
    field_evidences: Optional[List[Dict[str, Any]]] = None
) -> EligibilityDecision:
    """Convenience helper to evaluate loan eligibility."""
    engine = LoanEligibilityEngine(db=db)
    return engine.evaluate(
        application_id=application_id,
        loan_type=loan_type,
        document_slots=document_slots,
        extracted_fields=extracted_fields,
        validation_results=validation_results,
        cross_document_findings=cross_document_findings,
        risk_assessment=risk_assessment,
        field_evidences=field_evidences
    )
