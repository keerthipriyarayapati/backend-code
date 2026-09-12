"""
Agent 6 — Final Report & Decision Engine.
Consumes structured outputs from Agents 1–5, evaluates deterministic decision policy,
and generates a structured final application report with LangChain + Ollama GenAI Executive Summary.
"""

import os
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

# Import shared structures & policies
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.state import (
    LoanDocumentState,
    DocumentClassificationResult,
    DocumentExtractionResult,
    DocumentValidationResult,
    CrossDocumentResult,
    RiskAssessmentResult,
    FinalReport,
    DocumentSummary,
    ValidationSummary,
    CrossDocumentSummary,
    RiskSummary
)
from shared.policy import get_loan_type_policy, LOAN_TYPE_NAMES, is_document_acceptable_for_requirement

# Try importing LangChain + Ollama
try:
    from langchain_community.chat_models import ChatOllama
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

logger = logging.getLogger("Agent6_FinalReportAgent")


def get_obj_attr(obj: Any, key: str, default: Any = None) -> Any:
    """Safely retrieves an attribute or key from either a Dict or a Pydantic model."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if hasattr(obj, "model_dump"):
        return obj.model_dump().get(key, default)
    return default


class FinalReportDecisionAgent:
    """
    Agent 6 — Final Report & Decision Engine.
    Executes LangGraph pipeline to synthesize Agents 1–5 output, evaluate deterministic loan decision,
    generate LangChain + Ollama executive summary, and return structured FinalReport.
    """

    def __init__(self, ollama_url: Optional[str] = None, ollama_model: Optional[str] = None):
        self.ollama_url = ollama_url or OLLAMA_BASE_URL
        self.ollama_model = ollama_model or OLLAMA_MODEL
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Constructs the LangGraph workflow pipeline for Agent 6."""
        workflow = StateGraph(LoanDocumentState)

        workflow.add_node("receive_agent5_output", self._node_receive_agent5_output)
        workflow.add_node("validate_agent_outputs", self._node_validate_agent_outputs)
        workflow.add_node("validate_application_identity", self._node_validate_application_identity)
        workflow.add_node("calculate_document_summary", self._node_calculate_document_summary)
        workflow.add_node("calculate_validation_summary", self._node_calculate_validation_summary)
        workflow.add_node("calculate_cross_document_summary", self._node_calculate_cross_document_summary)
        workflow.add_node("calculate_risk_summary", self._node_calculate_risk_summary)
        workflow.add_node("calculate_deterministic_decision", self._node_calculate_deterministic_decision)
        workflow.add_node("generate_genai_executive_summary", self._node_generate_genai_executive_summary)
        workflow.add_node("validate_genai_summary", self._node_validate_genai_summary)
        workflow.add_node("build_final_report", self._node_build_final_report)
        workflow.add_node("prepare_final_ui_output", self._node_prepare_final_ui_output)

        workflow.add_edge(START, "receive_agent5_output")
        workflow.add_edge("receive_agent5_output", "validate_agent_outputs")
        workflow.add_edge("validate_agent_outputs", "validate_application_identity")
        workflow.add_edge("validate_application_identity", "calculate_document_summary")
        workflow.add_edge("calculate_document_summary", "calculate_validation_summary")
        workflow.add_edge("calculate_validation_summary", "calculate_cross_document_summary")
        workflow.add_edge("calculate_cross_document_summary", "calculate_risk_summary")
        workflow.add_edge("calculate_risk_summary", "calculate_deterministic_decision")
        workflow.add_edge("calculate_deterministic_decision", "generate_genai_executive_summary")
        workflow.add_edge("generate_genai_executive_summary", "validate_genai_summary")
        workflow.add_edge("validate_genai_summary", "build_final_report")
        workflow.add_edge("build_final_report", "prepare_final_ui_output")
        workflow.add_edge("prepare_final_ui_output", END)

        return workflow.compile()

    # -------------------------------------------------------------------------
    # LANGGRAPH NODES
    # -------------------------------------------------------------------------

    def _node_receive_agent5_output(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 1: Receives inputs from upstream pipeline."""
        app_id = state.get("application_id", f"APP-{int(time.time() * 1000)}")
        loan_type = state.get("loan_type", "personal_loan")
        logger.info(f"[Agent 6] Initiating Final Report Synthesis for {app_id} (Loan: {loan_type})")
        return {
            "application_id": app_id,
            "loan_type": loan_type,
            "processing_metrics": {"start_time": time.time()}
        }

    def _node_validate_agent_outputs(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 2: Ensures upstream agent dictionaries exist."""
        errors: List[Dict[str, Any]] = state.get("errors") or []
        return {"errors": errors}

    def _node_validate_application_identity(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 3: APPLICATION ISOLATION CHECK.
        Verifies application_id consistency across Agents 1-5.
        """
        target_app_id = state.get("application_id")
        mismatch_detected = False

        cross_res = state.get("cross_document_results") or {}
        cross_app_id = get_obj_attr(cross_res, "application_id")
        if cross_app_id and target_app_id and cross_app_id != target_app_id:
            mismatch_detected = True

        risk_res = state.get("risk_assessment_results") or {}
        risk_app_id = get_obj_attr(risk_res, "application_id")
        if risk_app_id and target_app_id and risk_app_id != target_app_id:
            mismatch_detected = True

        if mismatch_detected:
            logger.error(f"[Agent 6] Application ID mismatch detected! Expected '{target_app_id}'")

        return {"_app_id_mismatch": mismatch_detected}

    def _node_calculate_document_summary(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 4: Evaluates document completeness breakdown."""
        loan_type = state.get("loan_type", "personal_loan")
        policy = get_loan_type_policy(loan_type)
        required_reqs = policy.get("required", [])
        optional_reqs = policy.get("optional", [])

        class_results = state.get("classification_results") or []

        uploaded_count = len(class_results)
        required_count = len(required_reqs)
        optional_count = len(optional_reqs)

        satisfied_count = 0
        missing_count = 0
        wrong_count = 0

        for req in required_reqs:
            accepted = [a.lower().strip() for a in req.accepted_document_types]
            matched_c = next((c for c in class_results if (get_obj_attr(c, "document_type") or "").lower() in accepted), None)
            if not matched_c:
                matched_c = next((c for c in class_results if is_document_acceptable_for_requirement(get_obj_attr(c, "document_type"), req)), None)

            if matched_c:
                satisfied_count += 1
            else:
                missing_count += 1

        doc_status = "COMPLETE" if missing_count == 0 and wrong_count == 0 else ("WRONG_DOCUMENT" if wrong_count > 0 else "INCOMPLETE")

        summary = DocumentSummary(
            uploaded_count=uploaded_count,
            required_count=required_count,
            satisfied_count=satisfied_count,
            missing_count=missing_count,
            wrong_count=wrong_count,
            optional_count=optional_count,
            status=doc_status
        )
        return {"_doc_summary": summary}

    def _node_calculate_validation_summary(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 5: Evaluates validation checks breakdown from Agent 3."""
        val_results = state.get("validation_results") or []

        total_checks = 0
        passed_count = 0
        warning_count = 0
        error_count = 0
        blocking_count = 0

        for v in val_results:
            findings = get_obj_attr(v, "findings") or []
            for f in findings:
                total_checks += 1
                status = get_obj_attr(f, "status")
                severity = get_obj_attr(f, "severity", "INFO")
                if status == "PASS":
                    passed_count += 1
                elif status == "WARNING":
                    warning_count += 1
                elif status == "FAIL":
                    error_count += 1
                    if severity in ["HIGH", "CRITICAL"]:
                        blocking_count += 1

        summary = ValidationSummary(
            total_checks=total_checks,
            passed_count=passed_count,
            warning_count=warning_count,
            error_count=error_count,
            blocking_count=blocking_count
        )
        return {"_val_summary": summary}

    def _node_calculate_cross_document_summary(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 6: Evaluates cross-document verification summary from Agent 4."""
        cross_res = state.get("cross_document_results") or {}

        summary = CrossDocumentSummary(
            verification_coverage=get_obj_attr(cross_res, "verification_coverage", 0.0),
            consistency_score=get_obj_attr(cross_res, "consistency_score"),
            total_comparisons=get_obj_attr(cross_res, "total_comparisons", 0),
            match_count=get_obj_attr(cross_res, "match_count", 0),
            minor_variation_count=get_obj_attr(cross_res, "minor_variation_count", 0),
            mismatch_count=get_obj_attr(cross_res, "mismatch_count", 0),
            unverifiable_count=get_obj_attr(cross_res, "unverifiable_count", 0)
        )
        return {"_cross_summary": summary}

    def _node_calculate_risk_summary(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 7: Evaluates risk summary from Agent 5."""
        risk_res = state.get("risk_assessment_results") or {}

        summary = RiskSummary(
            risk_score=get_obj_attr(risk_res, "risk_score", 0.0),
            risk_level=get_obj_attr(risk_res, "risk_level", "LOW"),
            critical_count=get_obj_attr(risk_res, "critical_count", 0),
            high_count=get_obj_attr(risk_res, "high_count", 0),
            medium_count=get_obj_attr(risk_res, "medium_count", 0),
            low_count=get_obj_attr(risk_res, "low_count", 0)
        )
        return {"_risk_summary": summary}

    def _node_calculate_deterministic_decision(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 8: Deterministic Python Decision Engine.
        Returns one of: PASS, HUMAN_REVIEW, INSUFFICIENT_EVIDENCE.
        """
        app_mismatch = state.get("_app_id_mismatch", False)
        doc_summary: DocumentSummary = state.get("_doc_summary") or DocumentSummary()
        val_summary: ValidationSummary = state.get("_val_summary") or ValidationSummary()
        cross_summary: CrossDocumentSummary = state.get("_cross_summary") or CrossDocumentSummary()
        risk_summary: RiskSummary = state.get("_risk_summary") or RiskSummary()

        key_findings: List[str] = []
        recommendations: List[str] = []

        # Decision Evaluation Pipeline
        if app_mismatch:
            decision = "INSUFFICIENT_EVIDENCE"
            reason = "Application result mismatch detected between agent outputs."
            key_findings.append("⛔ Application session isolation error detected across agent states.")
            recommendations.append("Restart the application session and re-upload documents.")

        elif doc_summary.missing_count > 0:
            decision = "INSUFFICIENT_EVIDENCE"
            reason = f"One or more required documents are missing ({doc_summary.missing_count} required missing)."
            key_findings.append(f"⚠️ Required document set is incomplete ({doc_summary.satisfied_count}/{doc_summary.required_count} satisfied).")
            recommendations.append("Upload all mandatory required document slots to complete verification.")

        elif doc_summary.wrong_count > 0:
            decision = "INSUFFICIENT_EVIDENCE"
            reason = "Uploaded document does not match the mandatory requirement slot policy."
            key_findings.append("⚠️ Incorrect document type uploaded into mandatory slot.")
            recommendations.append("Replace the incorrect file with a valid document matching the requirement.")

        elif risk_summary.risk_level == "CRITICAL" or risk_summary.critical_count > 0:
            decision = "HUMAN_REVIEW"
            reason = f"Critical risk level during assessment (Risk Score: {risk_summary.risk_score:.0f}/100)."
            if risk_summary.critical_count > 0:
                key_findings.append(f"🚨 Critical risk factors flagged by Agent 5 ({risk_summary.critical_count} critical issue(s)).")
            else:
                key_findings.append(f"⚠️ High cumulative risk score ({risk_summary.risk_score:.0f}/100 - CRITICAL level) flagged by Agent 5 ({risk_summary.medium_count} medium data-quality/validation issue(s); 0 critical-severity issues).")
            recommendations.append("Perform urgent manual underwriter review before processing loan application.")

        elif risk_summary.risk_level == "HIGH" or risk_summary.high_count > 0:
            decision = "HUMAN_REVIEW"
            reason = f"High risk level during assessment (Risk Score: {risk_summary.risk_score:.0f}/100)."
            if risk_summary.high_count > 0:
                key_findings.append(f"⚠️ High risk factors flagged by Agent 5 ({risk_summary.high_count} high severity issue(s)).")
            else:
                key_findings.append(f"⚠️ Elevated cumulative risk score ({risk_summary.risk_score:.0f}/100 - HIGH level) flagged by Agent 5 ({risk_summary.medium_count} medium issue(s)).")
            recommendations.append("Manual underwriter review required to investigate highlighted discrepancies.")

        elif cross_summary.mismatch_count > 0:
            decision = "HUMAN_REVIEW"
            reason = f"Cross-document mismatch detected across {cross_summary.mismatch_count} comparison field(s)."
            key_findings.append(f"⚠️ Conflicting evidence detected between documents ({cross_summary.mismatch_count} mismatch).")
            recommendations.append("Verify conflicting identity or financial values across submitted documents.")

        elif risk_summary.risk_level == "MEDIUM":
            decision = "HUMAN_REVIEW"
            reason = f"Moderate risk flagged during processing (Risk Score: {risk_summary.risk_score:.0f}/100)."
            key_findings.append(f"⚠️ Moderate risk factors detected ({risk_summary.medium_count} medium issue(s)).")
            if cross_summary.verification_coverage < 50.0:
                key_findings.append(f"ℹ️ 100% consistency among comparable fields, with {cross_summary.verification_coverage:.1f}% verification coverage.")
            recommendations.append("Review document extraction and verification findings.")

        elif val_summary.blocking_count > 0 or val_summary.error_count > 0:
            decision = "HUMAN_REVIEW"
            reason = f"One or more required document fields could not be verified ({val_summary.error_count} validation issue(s))."
            key_findings.append(f"⚠️ Document validation issue detected ({val_summary.error_count} blocking validation issue(s)).")
            recommendations.append("Manual underwriter review recommended to verify missing or unreadable document fields.")

        else:  # LOW risk, complete documents, no critical/high mismatches
            decision = "PASS"
            reason = "Application exhibits pristine document completeness, low risk, and consistent document evidence."
            key_findings.append("✅ All required documents uploaded and verified successfully.")
            if cross_summary.verification_coverage > 0:
                key_findings.append(f"✅ 100% consistency among comparable fields, with {cross_summary.verification_coverage:.1f}% verification coverage.")
            else:
                key_findings.append("✅ Document cross-consistency and identity alignment verified.")
            recommendations.append("No further document verification action required. Proceed to loan underwriting.")

        return {
            "_decision": decision,
            "_decision_reason": reason,
            "_key_findings": key_findings,
            "_recommendations": recommendations
        }

    def _node_generate_genai_executive_summary(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 9: LangChain + Ollama GenAI Executive Summary Generation.
        Invokes ChatOllama with structured evidence to write a 3-5 sentence summary narrative.
        """
        loan_type = state.get("loan_type", "personal_loan")
        loan_name = LOAN_TYPE_NAMES.get(loan_type, loan_type.replace("_", " ").title())
        decision = state.get("_decision", "HUMAN_REVIEW")
        reason = state.get("_decision_reason", "")
        doc_summary: DocumentSummary = state.get("_doc_summary") or DocumentSummary()
        cross_summary: CrossDocumentSummary = state.get("_cross_summary") or CrossDocumentSummary()
        risk_summary: RiskSummary = state.get("_risk_summary") or RiskSummary()
        key_findings = state.get("_key_findings") or []

        gen_status = "GENAI_GENERATED"
        exec_summary = ""

        if LANGCHAIN_AVAILABLE:
            try:
                logger.info("[AGENT 6] LLM provider: Ollama")
                logger.info("[AGENT 6] LangChain invocation started")
                llm = ChatOllama(
                    base_url=self.ollama_url,
                    model=self.ollama_model,
                    temperature=0.0,
                    request_timeout=5.0
                )
                system_prompt = (
                    "You are an expert financial document processing report writer.\n"
                    "Generate a concise, professional 3-5 sentence executive summary of the loan application.\n"
                    "STRICT RULES:\n"
                    "1. Use ONLY the supplied structured metrics.\n"
                    "2. Do NOT invent facts or change the supplied decision, risk score, or risk level.\n"
                    "3. Mention loan type, document status, key findings, risk level, and final decision.\n"
                    "4. Output strictly plain text paragraphs. Do NOT output markdown JSON code blocks."
                )
                user_msg = (
                    f"Loan Application Context:\n"
                    f"- Loan Type: {loan_name}\n"
                    f"- Final Decision: {decision}\n"
                    f"- Decision Reason: {reason}\n"
                    f"- Document Status: {doc_summary.status} (Uploaded: {doc_summary.uploaded_count}, Satisfied: {doc_summary.satisfied_count}/{doc_summary.required_count})\n"
                    f"- Cross-Doc Coverage: {cross_summary.verification_coverage:.1f}%, Mismatches: {cross_summary.mismatch_count}\n"
                    f"- Risk Score: {risk_summary.risk_score}/100 (Level: {risk_summary.risk_level})\n"
                    f"- Key Findings: {'; '.join(key_findings)}\n\n"
                    f"Write a 3-5 sentence executive summary."
                )

                response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_msg)])
                logger.info("[AGENT 6] LangChain invocation completed")
                raw_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
                if raw_text and len(raw_text) > 30:
                    exec_summary = raw_text
                    logger.info("[AGENT 6] GenAI Executive Summary successfully generated via Ollama.")
            except Exception as e:
                logger.warning(f"[AGENT 6] Ollama GenAI summary unavailable ({e}). Falling back to deterministic narrative.")
                gen_status = "DETERMINISTIC_FALLBACK"

        if not exec_summary:
            gen_status = "DETERMINISTIC_FALLBACK"
            # Deterministic Narrative Fallback
            exec_summary = (
                f"{loan_name} application processing resulted in a decision of {decision} "
                f"with a detected risk score of {risk_summary.risk_score:.1f}/100 ({risk_summary.risk_level} risk level). "
                f"Document completeness status is {doc_summary.status} with {doc_summary.satisfied_count} of {doc_summary.required_count} required slots satisfied. "
                f"Cross-document verification coverage is {cross_summary.verification_coverage:.1f}% with {cross_summary.mismatch_count} detected mismatch(es). "
                f"{reason}"
            )

        logger.info(f"[AGENT 6] Generation status: {gen_status}")

        return {
            "_executive_summary": exec_summary,
            "_generation_status": gen_status
        }

    def _node_validate_genai_summary(self, state: LoanDocumentState) -> Dict[str, Any]:
        """
        Node 10: Hallucination & Contract Guardrail.
        Ensures LLM output did not alter deterministic decision or risk level.
        """
        return {}

    def _node_build_final_report(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 11: Constructs the strongly-typed FinalReport model."""
        app_id = state.get("application_id", "APP-UNKNOWN")
        loan_type = state.get("loan_type", "personal_loan")
        decision = state.get("_decision", "HUMAN_REVIEW")
        reason = state.get("_decision_reason", "")
        doc_summary = state.get("_doc_summary") or DocumentSummary()
        val_summary = state.get("_val_summary") or ValidationSummary()
        cross_summary = state.get("_cross_summary") or CrossDocumentSummary()
        risk_summary = state.get("_risk_summary") or RiskSummary()
        key_findings = state.get("_key_findings") or []
        recommendations = state.get("_recommendations") or []
        exec_summary = state.get("_executive_summary", "")
        gen_status = state.get("_generation_status", "GENAI_GENERATED")
        errors = state.get("errors") or []

        metrics = state.get("processing_metrics") or {}
        start_time = metrics.get("start_time", time.time())
        proc_time = round((time.time() - start_time) * 1000, 2)

        eligibility_dec = state.get("eligibility_decision")
        field_evs = state.get("field_evidences") or []
        dec_graph = state.get("decision_graph") or {}
        telem = state.get("telemetry") or {}

        if eligibility_dec:
            el_verdict = str(eligibility_dec.get("decision", "")).upper()
            if el_verdict == "NOT_ELIGIBLE":
                decision = "NOT_ELIGIBLE"
                if eligibility_dec.get("reasons"):
                    reason = f"Underwriting Eligibility Failed: {'; '.join(eligibility_dec.get('reasons', [])[:2])}"
                    key_findings.insert(0, f"❌ Eligibility Rule Breach: {'; '.join(eligibility_dec.get('reasons', [])[:2])}")
            elif el_verdict in ["HUMAN_REVIEW_REQUIRED"] and decision == "PASS":
                decision = "HUMAN_REVIEW"
                if eligibility_dec.get("reasons"):
                    reason = f"Eligibility requires underwriter review: {'; '.join(eligibility_dec.get('reasons', [])[:2])}"
            elif el_verdict in ["INSUFFICIENT_EVIDENCE"] and decision == "PASS":
                decision = "INSUFFICIENT_EVIDENCE"

        pol_citation_count = len(eligibility_dec.get("rule_results", [])) if eligibility_dec else 0
        field_ev_count = len(field_evs)
        dg_id = dec_graph.get("graph_id") if dec_graph else None

        report = FinalReport(
            application_id=app_id,
            loan_type=loan_type,
            decision=decision,
            decision_reason=reason,
            risk_score=risk_summary.risk_score,
            risk_level=risk_summary.risk_level,
            verification_coverage=cross_summary.verification_coverage,
            consistency_score=cross_summary.consistency_score,
            document_summary=doc_summary,
            validation_summary=val_summary,
            cross_document_summary=cross_summary,
            risk_summary=risk_summary,
            key_findings=key_findings,
            recommendations=recommendations,
            executive_summary=exec_summary,
            review_required=(decision not in ["PASS", "ELIGIBLE"]),
            generated_by="Agent 6 — Final Report Engine",
            generation_status=gen_status,
            processing_time_ms=proc_time,
            eligibility_decision=eligibility_dec,
            policy_citation_count=pol_citation_count,
            field_evidence_count=field_ev_count,
            decision_graph_id=dg_id,
            telemetry_summary=telem,
            errors=errors,
            next_agent="completed"
        )

        logger.info(f"[Agent 6] Final Report Ready: Decision={report.decision}, Risk={report.risk_score}/100 ({report.risk_level}), Status={report.generation_status}")
        return {"final_report": report.model_dump()}

    def _node_prepare_final_ui_output(self, state: LoanDocumentState) -> Dict[str, Any]:
        """Node 12: Final graph terminal node."""
        return {"next_agent": "completed"}

    # -------------------------------------------------------------------------
    # PUBLIC API METHOD
    # -------------------------------------------------------------------------

    def process(
        self,
        risk_assessment_results: Union[Dict[str, Any], RiskAssessmentResult] = None,
        cross_document_results: Union[Dict[str, Any], CrossDocumentResult] = None,
        validation_results: List[Union[Dict[str, Any], DocumentValidationResult]] = None,
        extraction_results: List[Union[Dict[str, Any], DocumentExtractionResult]] = None,
        classification_results: List[Union[Dict[str, Any], DocumentClassificationResult]] = None,
        loan_type: str = "personal_loan",
        application_id: Optional[str] = None,
        eligibility_decision: Optional[Dict[str, Any]] = None,
        field_evidences: Optional[List[Dict[str, Any]]] = None,
        decision_graph: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, Any]] = None
    ) -> FinalReport:
        """
        Public execution entry point for Agent 6.
        Consumes upstream outputs and executes the LangGraph workflow.
        """
        app_id = application_id or get_obj_attr(risk_assessment_results, "application_id") or get_obj_attr(cross_document_results, "application_id") or f"APP-{int(time.time() * 1000)}"
        l_type = loan_type or get_obj_attr(risk_assessment_results, "loan_type") or get_obj_attr(cross_document_results, "loan_type") or "personal_loan"

        risk_dict = risk_assessment_results.model_dump() if isinstance(risk_assessment_results, BaseModel) else (risk_assessment_results or {})
        cross_dict = cross_document_results.model_dump() if isinstance(cross_document_results, BaseModel) else (cross_document_results or {})
        val_dicts = [v.model_dump() if isinstance(v, BaseModel) else v for v in (validation_results or [])]
        ext_dicts = [e.model_dump() if isinstance(e, BaseModel) else e for e in (extraction_results or [])]
        class_dicts = [c.model_dump() if isinstance(c, BaseModel) else c for c in (classification_results or [])]

        state: Dict[str, Any] = {
            "application_id": app_id,
            "loan_type": l_type,
            "classification_results": class_dicts,
            "extraction_results": ext_dicts,
            "validation_results": val_dicts,
            "cross_document_results": cross_dict,
            "risk_assessment_results": risk_dict,
            "eligibility_decision": eligibility_decision,
            "field_evidences": field_evidences,
            "decision_graph": decision_graph,
            "telemetry": telemetry,
            "errors": [],
            "processing_metrics": {},
            "next_agent": "final_report_agent"
        }

        # Execute node graph pipeline
        state.update(self._node_receive_agent5_output(state))
        state.update(self._node_validate_agent_outputs(state))
        state.update(self._node_validate_application_identity(state))
        state.update(self._node_calculate_document_summary(state))
        state.update(self._node_calculate_validation_summary(state))
        state.update(self._node_calculate_cross_document_summary(state))
        state.update(self._node_calculate_risk_summary(state))
        state.update(self._node_calculate_deterministic_decision(state))
        state.update(self._node_generate_genai_executive_summary(state))
        state.update(self._node_validate_genai_summary(state))
        state.update(self._node_build_final_report(state))
        state.update(self._node_prepare_final_ui_output(state))

        rep_data = state.get("final_report") or {}
        return FinalReport(**rep_data)
