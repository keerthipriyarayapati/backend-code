"""
Evidence-Based Decision Graph Builder.
Constructs the 11-stage decision graph topology reflecting real pipeline execution and findings.
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session

from decision_graph.schemas import GraphNode, GraphEdge, DecisionGraphData
from database.repositories import (
    save_decision_graph,
    get_decision_graph_by_application
)

logger = logging.getLogger("DecisionGraphBuilder")


class DecisionGraphBuilder:
    """Constructs 11-stage decision graph from pipeline artifacts and agent results."""

    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def build(
        self,
        application_id: str,
        loan_type: str,
        document_slots: List[Dict[str, Any]],
        classifications: List[Dict[str, Any]],
        extracted_fields: Dict[str, Any],
        validation_results: List[Dict[str, Any]],
        cross_document_findings: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
        eligibility_decision: Optional[Dict[str, Any]] = None,
        final_report: Optional[Dict[str, Any]] = None,
        field_evidences: Optional[List[Dict[str, Any]]] = None
    ) -> DecisionGraphData:
        """Constructs and returns the complete 11-stage decision graph."""
        graph_id = f"DG_{application_id}_{uuid.uuid4().hex[:6]}"
        nodes: List[GraphNode] = []
        edges: List[GraphEdge] = []

        total_ev = len(field_evidences or [])
        classified_count = len(classifications)
        field_count = sum(len(v) if isinstance(v, dict) else 1 for v in extracted_fields.values()) if extracted_fields else 0

        # -------------------------------------------------------------
        # 1. Application Received
        # -------------------------------------------------------------
        nodes.append(GraphNode(
            node_id="app_received",
            stage_name="STAGE_1_APPLICATION_INTAKE",
            label="Application Intake & Registration",
            status="PASS",
            agent_name="System Gateway",
            description=f"Loan Application {application_id} successfully created and registered.",
            evidence_count=1,
            citation_count=1,
            node_metadata={"application_id": application_id, "loan_type": loan_type}
        ))

        # -------------------------------------------------------------
        # 2. Loan Type & Underwriting Policy Identified
        # -------------------------------------------------------------
        nodes.append(GraphNode(
            node_id="loan_type_identified",
            stage_name="STAGE_2_POLICY_RESOLUTION",
            label=f"{loan_type} Policy Identified",
            status="PASS" if loan_type else "WARNING",
            agent_name="Policy Knowledge Base",
            description=f"Mapped to active underwriting guidelines for '{loan_type}'.",
            evidence_count=1,
            citation_count=1,
            node_metadata={"loan_type": loan_type, "source_type": "DEMO_POLICY"}
        ))

        # -------------------------------------------------------------
        # 3. Mandatory Document Checklist
        # -------------------------------------------------------------
        missing_mandatory = [s.get("display_name", s.get("requirement_id")) for s in document_slots if s.get("required") and s.get("status") in ["pending", "missing"]]
        wrong_docs = [s.get("display_name", s.get("requirement_id")) for s in document_slots if s.get("status") in ["wrong_document", "rejected"]]
        
        doc_status = "PASS"
        if missing_mandatory or wrong_docs:
            doc_status = "FAIL"
        elif any(s.get("status") == "pending" for s in document_slots):
            doc_status = "WARNING"

        doc_desc = f"{len(document_slots)} document slots evaluated."
        if missing_mandatory:
            doc_desc += f" Missing required: {', '.join(missing_mandatory)}."
        if wrong_docs:
            doc_desc += f" Wrong documents: {', '.join(wrong_docs)}."

        nodes.append(GraphNode(
            node_id="required_docs_check",
            stage_name="STAGE_3_DOCUMENT_CHECKLIST",
            label="Document Requirement Checklist",
            status=doc_status,
            agent_name="Document Requirement Policy",
            description=doc_desc,
            evidence_count=len(document_slots),
            citation_count=len(document_slots),
            node_metadata={"missing_count": len(missing_mandatory), "wrong_count": len(wrong_docs)}
        ))

        # -------------------------------------------------------------
        # 4. Agent 1 Document Classification & OCR
        # -------------------------------------------------------------
        cls_failed = any(c.get("status") in ["failed", "unknown"] for c in classifications)
        cls_status = "PASS"
        if cls_failed:
            cls_status = "FAIL"
        elif any(float(c.get("confidence", 1.0)) < 0.70 for c in classifications):
            cls_status = "WARNING"

        nodes.append(GraphNode(
            node_id="agent_1_classify",
            stage_name="STAGE_4_AGENT_1_CLASSIFICATION",
            label="Agent 1: Document Classification & OCR",
            status=cls_status,
            agent_name="Agent 1 Document Classifier",
            description=f"Classified {classified_count} uploaded documents with optical character recognition fallback.",
            evidence_count=classified_count,
            citation_count=classified_count,
            node_metadata={"documents_classified": classified_count}
        ))

        # -------------------------------------------------------------
        # 5. Agent 2 Key-Value Field Extraction
        # -------------------------------------------------------------
        ext_status = "PASS" if field_count > 0 else "WARNING"
        nodes.append(GraphNode(
            node_id="agent_2_extract",
            stage_name="STAGE_5_AGENT_2_EXTRACTION",
            label="Agent 2: Key-Value Field & Evidence Extraction",
            status=ext_status,
            agent_name="Agent 2 Data Extractor",
            description=f"Extracted {field_count} structured fields with bounding source text snippets.",
            evidence_count=total_ev or field_count,
            citation_count=total_ev or field_count,
            node_metadata={"extracted_fields_count": field_count, "evidence_snippets_count": total_ev}
        ))

        # -------------------------------------------------------------
        # 6. Agent 3 Document Rule & Schema Validation
        # -------------------------------------------------------------
        val_fails = [v for v in validation_results if str(v.get("status", "")).upper() == "FAIL"]
        val_status = "PASS"
        if any(str(v.get("severity", "")).upper() in ["CRITICAL", "HIGH"] for v in val_fails):
            val_status = "FAIL"
        elif val_fails:
            val_status = "WARNING"

        nodes.append(GraphNode(
            node_id="agent_3_validate",
            stage_name="STAGE_6_AGENT_3_VALIDATION",
            label="Agent 3: Document Rule & Schema Validation",
            status=val_status,
            agent_name="Agent 3 Data Validator",
            description=f"Validated field patterns and dates. {len(val_fails)} validation failure(s) recorded.",
            evidence_count=len(validation_results),
            citation_count=len(validation_results),
            node_metadata={"total_validations": len(validation_results), "failures": len(val_fails)}
        ))

        # -------------------------------------------------------------
        # 7. Agent 4 Cross-Document Reconciliation
        # -------------------------------------------------------------
        cross_mismatches = [c for c in cross_document_findings if str(c.get("match_status") or c.get("status") or "").upper() == "MISMATCH"]
        cross_status = "PASS"
        if any(str(c.get("severity", "")).upper() in ["CRITICAL", "HIGH"] for c in cross_mismatches):
            cross_status = "FAIL"
        elif cross_mismatches:
            cross_status = "WARNING"

        nodes.append(GraphNode(
            node_id="agent_4_cross_doc",
            stage_name="STAGE_7_AGENT_4_CROSS_DOC",
            label="Agent 4: Cross-Document Reconciliation",
            status=cross_status,
            agent_name="Agent 4 Cross-Document Reconciler",
            description=f"Cross-verified names, PANs, and dates across multiple files. {len(cross_mismatches)} mismatch(es) found.",
            evidence_count=len(cross_document_findings),
            citation_count=len(cross_document_findings),
            node_metadata={"comparisons_count": len(cross_document_findings), "mismatches": len(cross_mismatches)}
        ))

        # -------------------------------------------------------------
        # 8. Agent 5 Comprehensive Risk Assessment
        # -------------------------------------------------------------
        r_score = float(risk_assessment.get("risk_score", 0.0) or 0.0)
        r_level = str(risk_assessment.get("risk_level", "LOW")).upper()
        risk_status = "PASS"
        if r_score >= 50.0 or r_level in ["HIGH", "CRITICAL"]:
            risk_status = "FAIL"
        elif r_score >= 25.0 or r_level in ["MEDIUM", "MODERATE"]:
            risk_status = "WARNING"

        nodes.append(GraphNode(
            node_id="agent_5_risk",
            stage_name="STAGE_8_AGENT_5_RISK",
            label="Agent 5: Risk & Fraud Assessment",
            status=risk_status,
            agent_name="Agent 5 Risk Assessor",
            description=f"Synthesized weighted risk score: {r_score:.1f}/100 (Level: {r_level}).",
            evidence_count=len(risk_assessment.get("risk_factors", [])),
            citation_count=len(risk_assessment.get("risk_factors", [])),
            node_metadata={"risk_score": r_score, "risk_level": r_level}
        ))

        # -------------------------------------------------------------
        # 9. Policy Evaluation (Policy KB)
        # -------------------------------------------------------------
        el_rule_results = eligibility_decision.get("rule_results", []) if eligibility_decision else []
        failed_policy_rules = [r for r in el_rule_results if r.get("status") == "FAIL"]
        policy_status = "PASS"
        if failed_policy_rules:
            policy_status = "FAIL"
        elif any(r.get("status") == "INSUFFICIENT_EVIDENCE" for r in el_rule_results):
            policy_status = "WARNING"

        nodes.append(GraphNode(
            node_id="policy_evaluation",
            stage_name="STAGE_9_POLICY_EVALUATION",
            label="Policy Knowledge Base Evaluation",
            status=policy_status,
            agent_name="Loan Policy Knowledge Base",
            description=f"Evaluated underwriting criteria. {len(failed_policy_rules)} rule(s) breached.",
            evidence_count=len(el_rule_results),
            citation_count=len(el_rule_results),
            node_metadata={"rules_evaluated": len(el_rule_results), "rules_failed": len(failed_policy_rules)}
        ))

        # -------------------------------------------------------------
        # 10. Loan Eligibility Engine
        # -------------------------------------------------------------
        el_verdict = str(eligibility_decision.get("decision", "INSUFFICIENT_EVIDENCE") if eligibility_decision else "INSUFFICIENT_EVIDENCE").upper()
        el_status = "PASS"
        if el_verdict == "ELIGIBLE":
            el_status = "PASS"
        elif el_verdict == "NOT_ELIGIBLE":
            el_status = "FAIL"
        elif el_verdict in ["HUMAN_REVIEW_REQUIRED", "INSUFFICIENT_EVIDENCE"]:
            el_status = "WARNING"

        nodes.append(GraphNode(
            node_id="eligibility_engine",
            stage_name="STAGE_10_ELIGIBILITY_ENGINE",
            label="Loan Eligibility Engine",
            status=el_status,
            agent_name="Loan Eligibility Engine",
            description=f"Eligibility verdict: {el_verdict}. {'; '.join(eligibility_decision.get('reasons', [])[:2]) if eligibility_decision else ''}",
            evidence_count=len(eligibility_decision.get("reasons", [])) if eligibility_decision else 0,
            citation_count=len(el_rule_results),
            node_metadata={"decision": el_verdict}
        ))

        # -------------------------------------------------------------
        # 11. Agent 6 Final Audit Report & Exporter
        # -------------------------------------------------------------
        final_dec = str(final_report.get("decision", el_verdict) if final_report else el_verdict).upper()
        f_status = "PASS"
        if final_dec in ["PASS", "ELIGIBLE"]:
            f_status = "PASS"
        elif final_dec in ["FAIL", "REJECT", "NOT_ELIGIBLE"]:
            f_status = "FAIL"
        else:
            f_status = "WARNING"

        nodes.append(GraphNode(
            node_id="agent_6_final_decision",
            stage_name="STAGE_11_AGENT_6_FINAL_REPORT",
            label="Agent 6: Final Underwriting Decision & Report",
            status=f_status,
            agent_name="Agent 6 Report Generator",
            description=f"Final consolidated outcome: {final_dec}. Comprehensive audit report and PDF dossier generated.",
            evidence_count=1,
            citation_count=1,
            node_metadata={"final_decision": final_dec}
        ))

        # -------------------------------------------------------------
        # Directed Edges (10 Sequential Transitions)
        # -------------------------------------------------------------
        edge_defs = [
            ("app_received", "loan_type_identified", "Registered"),
            ("loan_type_identified", "required_docs_check", "Policy Loaded"),
            ("required_docs_check", "agent_1_classify", "Uploads Received"),
            ("agent_1_classify", "agent_2_extract", "Classified & OCR"),
            ("agent_2_extract", "agent_3_validate", "Fields Extracted"),
            ("agent_3_validate", "agent_4_cross_doc", "Validated"),
            ("agent_4_cross_doc", "agent_5_risk", "Reconciled"),
            ("agent_5_risk", "policy_evaluation", "Risk Assessed"),
            ("policy_evaluation", "eligibility_engine", "Rules Tested"),
            ("eligibility_engine", "agent_6_final_decision", "Decision Synthesized")
        ]

        for src, tgt, lbl in edge_defs:
            edges.append(GraphEdge(source=src, target=tgt, label=lbl, type="DIRECTED"))

        # Aggregate counts
        passed_cnt = sum(1 for n in nodes if n.status == "PASS")
        failed_cnt = sum(1 for n in nodes if n.status == "FAIL")
        warning_cnt = sum(1 for n in nodes if n.status == "WARNING")
        info_cnt = sum(1 for n in nodes if n.status == "INFO")

        graph_data = DecisionGraphData(
            graph_id=graph_id,
            application_id=application_id,
            total_nodes=len(nodes),
            total_edges=len(edges),
            passed_nodes=passed_cnt,
            failed_nodes=failed_cnt,
            warning_nodes=warning_cnt,
            info_nodes=info_cnt,
            nodes=nodes,
            edges=edges
        )

        # Persist to database if db session provided
        if self.db:
            try:
                g_dict = {
                    "graph_id": graph_data.graph_id,
                    "application_id": graph_data.application_id,
                    "total_nodes": graph_data.total_nodes,
                    "total_edges": graph_data.total_edges,
                    "passed_nodes": graph_data.passed_nodes,
                    "failed_nodes": graph_data.failed_nodes,
                    "warning_nodes": graph_data.warning_nodes,
                    "info_nodes": graph_data.info_nodes
                }
                n_dicts = [n.dict() for n in graph_data.nodes]
                e_dicts = [
                    {
                        "source_node_id": e.source,
                        "target_node_id": e.target,
                        "edge_label": e.label,
                        "edge_type": e.type
                    } for e in graph_data.edges
                ]
                save_decision_graph(self.db, g_dict, n_dicts, e_dicts)
                logger.info(f"Persisted decision graph {graph_id} for {application_id}")
            except Exception as e:
                logger.error(f"Failed to persist decision graph: {e}", exc_info=True)

        return graph_data


def build_application_decision_graph(
    db: Optional[Session],
    application_id: str,
    loan_type: str,
    document_slots: List[Dict[str, Any]],
    classifications: List[Dict[str, Any]],
    extracted_fields: Dict[str, Any],
    validation_results: List[Dict[str, Any]],
    cross_document_findings: List[Dict[str, Any]],
    risk_assessment: Dict[str, Any],
    eligibility_decision: Optional[Dict[str, Any]] = None,
    final_report: Optional[Dict[str, Any]] = None,
    field_evidences: Optional[List[Dict[str, Any]]] = None
) -> DecisionGraphData:
    """Convenience helper to build decision graph."""
    builder = DecisionGraphBuilder(db=db)
    return builder.build(
        application_id=application_id,
        loan_type=loan_type,
        document_slots=document_slots,
        classifications=classifications,
        extracted_fields=extracted_fields,
        validation_results=validation_results,
        cross_document_findings=cross_document_findings,
        risk_assessment=risk_assessment,
        eligibility_decision=eligibility_decision,
        final_report=final_report,
        field_evidences=field_evidences
    )
