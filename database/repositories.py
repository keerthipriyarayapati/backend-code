"""
Database Repositories for Loan Document Processing AI System.
Provides structured data access and persistence methods for all 12 database tables.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func, desc, asc, or_

from database.models import (
    LoanApplication,
    Applicant,
    DocumentRequirementModel,
    DocumentModel,
    DocumentClassificationModel,
    ExtractedFieldModel,
    ValidationResultModel,
    CrossDocumentFindingModel,
    RiskAssessmentModel,
    RiskFactorModel,
    FinalReportModel,
    ProcessingRunModel,
    LoanPolicyModel,
    PolicyRuleModel,
    PolicyRequiredDocumentModel,
    PolicyRequiredFieldModel,
    FieldEvidenceModel,
    DecisionGraphModel,
    DecisionGraphNodeModel,
    DecisionGraphEdgeModel,
    EligibilityResultModel,
    EligibilityRuleResultModel,
    AgentExecutionLogModel,
    TelemetryMetricsModel,
    User
)

logger = logging.getLogger("DatabaseRepositories")


# =============================================================================
# 1. LOAN APPLICATIONS REPOSITORY
# =============================================================================

def create_application(
    db: Session,
    application_id: str,
    loan_type: str,
    applicant_name: Optional[str] = None,
    status: str = "NOT_STARTED",
    employee_id: Optional[str] = None,
    branch_id: Optional[str] = None
) -> LoanApplication:
    """Creates a new loan application record."""
    existing = db.query(LoanApplication).filter(LoanApplication.application_id == application_id).first()
    if existing:
        return existing

    app = LoanApplication(
        application_id=application_id,
        loan_type=loan_type,
        applicant_name=applicant_name,
        status=status,
        employee_id=employee_id,
        branch_id=branch_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(app)
    db.commit()
    db.refresh(app)

    # Automatically create primary applicant record
    create_applicant(
        db,
        application_id=application_id,
        applicant_type="PRIMARY_APPLICANT",
        full_name=applicant_name or "Primary Applicant"
    )

    return app


def get_application(db: Session, application_id: str) -> Optional[LoanApplication]:
    """Retrieves a loan application by application_id."""
    return db.query(LoanApplication).filter(LoanApplication.application_id == application_id).first()


def list_applications(db: Session, limit: int = 50) -> List[LoanApplication]:
    """Lists loan applications."""
    return db.query(LoanApplication).order_by(LoanApplication.created_at.desc()).limit(limit).all()


def update_application_status(db: Session, application_id: str, status: str) -> Optional[LoanApplication]:
    """Updates loan application status."""
    app = get_application(db, application_id)
    if app:
        app.status = status
        app.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(app)
    return app


def update_application_completion(
    db: Session,
    application_id: str,
    status: str,
    risk_level: Optional[str] = None,
    processing_time: Optional[float] = None
) -> Optional[LoanApplication]:
    """Updates loan application completion metadata when final report finishes."""
    app = get_application(db, application_id)
    if app:
        app.status = status
        if risk_level:
            app.risk_level = risk_level
        if processing_time is not None:
            app.processing_time = processing_time
        app.completed_at = datetime.utcnow()
        app.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(app)
    return app



# =============================================================================
# 2. APPLICANTS REPOSITORY
# =============================================================================

def create_applicant(
    db: Session,
    application_id: str,
    applicant_type: str = "PRIMARY_APPLICANT",
    full_name: Optional[str] = None,
    date_of_birth: Optional[str] = None,
    gender: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
    address: Optional[str] = None
) -> Applicant:
    """Creates an applicant record for a loan application."""
    applicant = Applicant(
        application_id=application_id,
        applicant_type=applicant_type,
        full_name=full_name,
        date_of_birth=date_of_birth,
        gender=gender,
        email=email,
        phone=phone,
        address=address,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(applicant)
    db.commit()
    db.refresh(applicant)
    return applicant


def get_applicants(db: Session, application_id: str) -> List[Applicant]:
    """Retrieves all applicants for an application_id."""
    return db.query(Applicant).filter(Applicant.application_id == application_id).all()


# =============================================================================
# 3. DOCUMENT REQUIREMENTS REPOSITORY
# =============================================================================

def get_document_requirements(db: Session, loan_type: str) -> List[DocumentRequirementModel]:
    """Retrieves required document slots for a loan type from database."""
    reqs = db.query(DocumentRequirementModel).filter(
        DocumentRequirementModel.loan_type == loan_type.lower().strip()
    ).all()
    if not reqs:
        # Fallback query
        reqs = db.query(DocumentRequirementModel).filter(
            DocumentRequirementModel.loan_type == "personal_loan"
        ).all()
    return reqs


# =============================================================================
# 4. DOCUMENTS REPOSITORY
# =============================================================================

def create_document(
    db: Session,
    application_id: str,
    file_name: str,
    file_path: str,
    requirement_id: Optional[str] = None,
    applicant_id: Optional[int] = None,
    file_extension: Optional[str] = None,
    mime_type: Optional[str] = None,
    document_type: Optional[str] = None,
    upload_status: str = "accepted",
    extraction_method: Optional[str] = None,
    ocr_used: bool = False,
    ocr_success: bool = False,
    extraction_error: Optional[str] = None,
    is_active: bool = True,
    text_quality: Optional[str] = None
) -> DocumentModel:
    """Creates a document metadata record in SQLite, deactivating previous active records in the slot."""
    # Ensure application exists
    app = get_application(db, application_id)
    if not app:
        create_application(db, application_id=application_id, loan_type="personal_loan")

    # If applicant_id not provided, assign primary applicant ID
    if not applicant_id:
        applicants = get_applicants(db, application_id)
        if applicants:
            applicant_id = applicants[0].id

    # Enforce single active document per (application_id, requirement_id)
    if requirement_id and is_active:
        prev_docs = db.query(DocumentModel).filter(
            DocumentModel.application_id == application_id,
            DocumentModel.requirement_id == requirement_id,
            DocumentModel.is_active == True
        ).all()
        for prev in prev_docs:
            prev.is_active = False
            prev.updated_at = datetime.utcnow()

    doc = DocumentModel(
        application_id=application_id,
        applicant_id=applicant_id,
        requirement_id=requirement_id,
        file_name=file_name,
        file_path=file_path,
        file_extension=file_extension or (file_name.rsplit('.', 1)[-1] if '.' in file_name else ""),
        mime_type=mime_type or "application/octet-stream",
        document_type=document_type,
        upload_status=upload_status,
        extraction_method=extraction_method,
        ocr_used=ocr_used,
        ocr_success=ocr_success,
        extraction_error=extraction_error,
        is_active=is_active,
        text_quality=text_quality,
        uploaded_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_documents(db: Session, application_id: str, active_only: bool = True) -> List[DocumentModel]:
    """Retrieves all documents associated with an application_id. Defaults to active only."""
    query = db.query(DocumentModel).filter(DocumentModel.application_id == application_id)
    if active_only:
        query = query.filter(DocumentModel.is_active == True)
    return query.all()


def get_document_by_id(db: Session, document_id: int) -> Optional[DocumentModel]:
    """Retrieves a document by PK integer ID."""
    return db.query(DocumentModel).filter(DocumentModel.id == document_id).first()


def delete_document(db: Session, document_id: int) -> bool:
    """Deletes a document record by ID."""
    doc = get_document_by_id(db, document_id)
    if doc:
        db.delete(doc)
        db.commit()
        return True
    return False


# =============================================================================
# 5. DOCUMENT CLASSIFICATION REPOSITORY (Agent 1 Output)
# =============================================================================

def save_classification(
    db: Session,
    document_id: int,
    predicted_document_type: str,
    confidence: float,
    classification_status: str = "CLASSIFIED",
    loan_type: Optional[str] = None,
    classification_reason: Optional[str] = None,
    model_name: str = "Agent1_DocumentClassifier"
) -> DocumentClassificationModel:
    """Saves Agent 1 document classification output."""
    cls_rec = DocumentClassificationModel(
        document_id=document_id,
        predicted_document_type=predicted_document_type,
        confidence=confidence,
        classification_status=classification_status,
        loan_type=loan_type,
        classification_reason=classification_reason,
        model_name=model_name,
        created_at=datetime.utcnow()
    )
    db.add(cls_rec)
    
    # Also update document_type on document table
    doc = get_document_by_id(db, document_id)
    if doc:
        doc.document_type = predicted_document_type
        doc.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(cls_rec)
    return cls_rec


def get_classifications_by_application(db: Session, application_id: str) -> List[Dict[str, Any]]:
    """Retrieves Agent 1 classification results for an application."""
    docs = get_documents(db, application_id)
    results = []
    for doc in docs:
        for cls in doc.classifications:
            results.append({
                "document_id": str(doc.id),
                "filename": doc.file_name,
                "file_extension": f".{doc.file_extension}" if doc.file_extension and not doc.file_extension.startswith('.') else (doc.file_extension or ""),
                "file_path": doc.file_path,
                "document_type": cls.predicted_document_type,
                "confidence": cls.confidence,
                "classification_reason": cls.classification_reason or "",
                "status": cls.classification_status,
                "requirement_id": doc.requirement_id
            })
    return results


# =============================================================================
# 6. EXTRACTED FIELDS REPOSITORY (Agent 2 Output)
# =============================================================================

def save_extracted_fields(
    db: Session,
    document_id: int,
    fields: Dict[str, Any],
    extraction_method: Optional[str] = None
) -> List[ExtractedFieldModel]:
    """Saves Agent 2 extracted key-value fields."""
    saved_records = []
    for field_name, val_info in fields.items():
        val = None
        conf = 0.0
        src_page = 1
        src_text = None
        data_type = "string"

        if isinstance(val_info, dict):
            val = str(val_info.get("value")) if val_info.get("value") is not None else None
            conf = float(val_info.get("confidence", 0.0))
            src = val_info.get("source", {})
            if isinstance(src, dict):
                src_page = src.get("page", 1)
                src_text = src.get("snippet") or src.get("raw_text") or src.get("text")
        else:
            val = str(val_info) if val_info is not None else None
            conf = 1.0

        rec = ExtractedFieldModel(
            document_id=document_id,
            field_name=field_name,
            field_value=val,
            normalized_value=val.lower().strip() if val else None,
            data_type=data_type,
            confidence=conf,
            source_page=src_page,
            source_text=src_text,
            extraction_method=extraction_method or "Regex / Key-Value Extractor",
            created_at=datetime.utcnow()
        )
        db.add(rec)
        saved_records.append(rec)

    # Automatically synchronize to field_evidence table
    try:
        from agents.agent_5_risk.agent import mask_sensitive_value
        doc = get_document_by_id(db, document_id)
        if doc and doc.application_id:
            for rec in saved_records:
                if rec.field_value:
                    ev_id = f"EV_{doc.application_id}_{rec.field_name}_{document_id}"
                    existing_ev = db.query(FieldEvidenceModel).filter(FieldEvidenceModel.evidence_id == ev_id).first()
                    masked_val = mask_sensitive_value(rec.field_value, rec.field_name)
                    if existing_ev:
                        existing_ev.extracted_value = masked_val
                        existing_ev.raw_value = masked_val
                        existing_ev.snippet = rec.source_text
                        existing_ev.source_page = rec.source_page
                        existing_ev.confidence = rec.confidence
                    else:
                        ev_item = FieldEvidenceModel(
                            evidence_id=ev_id,
                            application_id=doc.application_id,
                            document_id=document_id,
                            slot_id=doc.requirement_id,
                            document_type=doc.document_type,
                            field_name=rec.field_name,
                            extracted_value=masked_val,
                            raw_value=masked_val,
                            normalized_value=rec.normalized_value,
                            source_page=rec.source_page,
                            snippet=rec.source_text,
                            extraction_method=rec.extraction_method,
                            ocr_used=bool(doc.ocr_used),
                            confidence=rec.confidence,
                            validation_status="EXTRACTED",
                            created_at=datetime.utcnow()
                        )
                        db.add(ev_item)
    except Exception as e:
        logger.warning(f"Error synchronizing field evidence for doc {document_id}: {e}")

    db.commit()
    return saved_records


def get_extraction_results_by_application(db: Session, application_id: str) -> List[Dict[str, Any]]:
    """Retrieves Agent 2 extraction results for an application."""
    docs = get_documents(db, application_id)
    results = []
    for doc in docs:
        fields_dict = {}
        for ef in doc.extracted_fields:
            fields_dict[ef.field_name] = {
                "value": ef.field_value,
                "confidence": ef.confidence,
                "source": {"page": ef.source_page, "snippet": ef.source_text}
            }
        if fields_dict:
            results.append({
                "document_id": str(doc.id),
                "filename": doc.file_name,
                "document_type": doc.document_type or "unknown",
                "fields": fields_dict,
                "overall_confidence": sum(f["confidence"] for f in fields_dict.values()) / max(len(fields_dict), 1),
                "extraction_status": "success"
            })
    return results


# =============================================================================
# 7. VALIDATION RESULTS REPOSITORY (Agent 3 Output)
# =============================================================================

def save_validation_results(
    db: Session,
    application_id: str,
    validation_results_list: List[Dict[str, Any]]
) -> List[ValidationResultModel]:
    """Saves Agent 3 validation findings."""
    saved_records = []
    for vr in validation_results_list:
        doc_id_val = vr.get("document_id")
        doc_pk = None
        if doc_id_val and str(doc_id_val).isdigit():
            doc_pk = int(doc_id_val)

        findings = vr.get("findings", [])
        for f in findings:
            rec = ValidationResultModel(
                application_id=application_id,
                document_id=doc_pk,
                field_name=f.get("field_name", "general"),
                validation_type=f.get("finding_type", "FIELD_VALIDATION"),
                status=f.get("status", "PASS"),
                severity=f.get("severity", "INFO"),
                expected_value=str(f.get("expected_value")) if f.get("expected_value") else None,
                actual_value=str(f.get("value")) if f.get("value") is not None else None,
                explanation=f.get("message", ""),
                evidence=json.dumps(f.get("source")) if f.get("source") else None,
                created_at=datetime.utcnow()
            )
            db.add(rec)
            saved_records.append(rec)

    db.commit()
    return saved_records


def get_validation_results_by_application(db: Session, application_id: str) -> List[Dict[str, Any]]:
    """Retrieves Agent 3 validation results for an application."""
    records = db.query(ValidationResultModel).filter(ValidationResultModel.application_id == application_id).all()
    grouped: Dict[Optional[int], List[ValidationResultModel]] = {}
    for r in records:
        grouped.setdefault(r.document_id, []).append(r)

    results = []
    for doc_id, findings_recs in grouped.items():
        doc = get_document_by_id(db, doc_id) if doc_id else None
        findings = []
        for f in findings_recs:
            findings.append({
                "field_name": f.field_name,
                "status": f.status,
                "finding_type": f.validation_type,
                "message": f.explanation or "",
                "severity": f.severity,
                "value": f.actual_value
            })
        results.append({
            "document_id": str(doc_id) if doc_id else "app_wide",
            "filename": doc.file_name if doc else "Application Validation",
            "document_type": doc.document_type if doc else "application",
            "validation_status": "PASS" if all(f["status"] == "PASS" for f in findings) else "WARNING",
            "findings": findings
        })
    return results


# =============================================================================
# 8. CROSS-DOCUMENT FINDINGS REPOSITORY (Agent 4 Output)
# =============================================================================

def save_cross_document_findings(
    db: Session,
    application_id: str,
    cross_doc_dict: Dict[str, Any]
) -> List[CrossDocumentFindingModel]:
    """Saves Agent 4 cross-document findings."""
    saved_records = []
    findings = cross_doc_dict.get("findings", [])
    for f in findings:
        rec = CrossDocumentFindingModel(
            application_id=application_id,
            field=f.get("field_name", "general"),
            category=f.get("doc1_type", "CROSS_DOC"),
            severity=f.get("severity", "INFO"),
            value_a=str(f.get("doc1_value")) if f.get("doc1_value") is not None else None,
            value_b=str(f.get("doc2_value")) if f.get("doc2_value") is not None else None,
            normalized_value_a=str(f.get("doc1_value")).lower().strip() if f.get("doc1_value") is not None else None,
            normalized_value_b=str(f.get("doc2_value")).lower().strip() if f.get("doc2_value") is not None else None,
            comparison_result=f.get("comparison_result", "UNVERIFIABLE"),
            explanation=f.get("message", ""),
            confidence=float(f.get("confidence", 1.0)),
            evidence=json.dumps(f.get("source")) if f.get("source") else None,
            created_at=datetime.utcnow()
        )
        db.add(rec)
        saved_records.append(rec)

    db.commit()
    return saved_records


def get_cross_document_findings_by_application(db: Session, application_id: str) -> Dict[str, Any]:
    """Retrieves Agent 4 cross-document findings for an application."""
    records = db.query(CrossDocumentFindingModel).filter(CrossDocumentFindingModel.application_id == application_id).all()
    findings = []
    for r in records:
        findings.append({
            "field_name": r.field,
            "doc1_type": r.category or "Source 1",
            "doc1_value": r.value_a,
            "doc2_type": "Source 2",
            "doc2_value": r.value_b,
            "comparison_result": r.comparison_result,
            "severity": r.severity,
            "message": r.explanation or "",
            "confidence": r.confidence
        })

    match_cnt = sum(1 for f in findings if f["comparison_result"] == "MATCH")
    mismatch_cnt = sum(1 for f in findings if f["comparison_result"] == "MISMATCH")
    unverifiable_cnt = sum(1 for f in findings if f["comparison_result"] == "UNVERIFIABLE")
    minor_cnt = sum(1 for f in findings if f["comparison_result"] == "MINOR_VARIATION")

    return {
        "application_id": application_id,
        "loan_type": "personal_loan",
        "total_comparisons": len(findings),
        "match_count": match_cnt,
        "mismatch_count": mismatch_cnt,
        "unverifiable_count": unverifiable_cnt,
        "minor_variation_count": minor_cnt,
        "findings": findings
    }


# =============================================================================
# 9. RISK ASSESSMENTS REPOSITORY (Agent 5 Output)
# =============================================================================

def save_risk_assessment(
    db: Session,
    application_id: str,
    risk_dict: Dict[str, Any]
) -> RiskAssessmentModel:
    """Saves Agent 5 risk assessment and risk factors."""
    assessment = RiskAssessmentModel(
        application_id=application_id,
        risk_score=float(risk_dict.get("risk_score", 0.0)),
        risk_level=risk_dict.get("risk_level", "LOW"),
        recommended_action=risk_dict.get("recommended_action", "PROCEED_TO_REPORT"),
        overall_reason=risk_dict.get("overall_summary", ""),
        created_at=datetime.utcnow()
    )
    db.add(assessment)
    db.commit()
    db.refresh(assessment)

    factors = risk_dict.get("risk_factors", [])
    for rf in factors:
        factor_rec = RiskFactorModel(
            risk_assessment_id=assessment.id,
            category=rf.get("category", "OTHER"),
            severity=rf.get("severity", "LOW"),
            points=float(rf.get("points", 0.0)),
            title=rf.get("title", "Risk Factor"),
            description=rf.get("description", ""),
            source=rf.get("source", "Agent 5 Risk Engine"),
            created_at=datetime.utcnow()
        )
        db.add(factor_rec)

    db.commit()
    return assessment


def get_risk_assessment_by_application(db: Session, application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves Agent 5 risk assessment for an application."""
    assessment = db.query(RiskAssessmentModel).filter(
        RiskAssessmentModel.application_id == application_id
    ).order_by(RiskAssessmentModel.created_at.desc()).first()

    if not assessment:
        return None

    factors = []
    for rf in assessment.risk_factors:
        factors.append({
            "factor_id": str(rf.id),
            "category": rf.category,
            "severity": rf.severity,
            "title": rf.title,
            "description": rf.description or "",
            "source": rf.source,
            "points": rf.points
        })

    return {
        "application_id": application_id,
        "loan_type": assessment.application.loan_type if assessment.application else "personal_loan",
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "recommended_action": assessment.recommended_action,
        "overall_summary": assessment.overall_reason or "",
        "total_risk_factors": len(factors),
        "risk_factors": factors
    }


# =============================================================================
# 10. FINAL REPORTS REPOSITORY (Agent 6 Output)
# =============================================================================

def save_final_report(
    db: Session,
    application_id: str,
    report_dict: Dict[str, Any]
) -> FinalReportModel:
    """Saves Agent 6 final report."""
    rec = FinalReportModel(
        application_id=application_id,
        decision=report_dict.get("decision", "PASS"),
        decision_reason=report_dict.get("decision_reason", ""),
        risk_score=float(report_dict.get("risk_score", 0.0)),
        risk_level=report_dict.get("risk_level", "LOW"),
        verification_coverage=float(report_dict.get("verification_coverage", 0.0)),
        consistency_score=float(report_dict.get("consistency_score")) if report_dict.get("consistency_score") is not None else None,
        document_summary=json.dumps(report_dict.get("document_summary", {})),
        validation_summary=json.dumps(report_dict.get("validation_summary", {})),
        cross_document_summary=json.dumps(report_dict.get("cross_document_summary", {})),
        risk_summary=json.dumps(report_dict.get("risk_summary", {})),
        key_findings=json.dumps(report_dict.get("key_findings", [])),
        recommendations=json.dumps(report_dict.get("recommendations", [])),
        executive_summary=report_dict.get("executive_summary", ""),
        review_required=bool(report_dict.get("review_required", False)),
        generated_by=report_dict.get("generated_by", "Agent 6 — Final Report Engine"),
        generation_status=report_dict.get("generation_status", "GENAI_GENERATED"),
        processing_time_ms=float(report_dict.get("processing_time_ms", 0.0)),
        created_at=datetime.utcnow()
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_final_report_by_application(db: Session, application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves Agent 6 final report for an application."""
    rep = db.query(FinalReportModel).filter(
        FinalReportModel.application_id == application_id
    ).order_by(FinalReportModel.created_at.desc()).first()

    if not rep:
        return None

    return {
        "application_id": rep.application_id,
        "loan_type": rep.application.loan_type if rep.application else "personal_loan",
        "decision": rep.decision,
        "decision_reason": rep.decision_reason or "",
        "risk_score": rep.risk_score,
        "risk_level": rep.risk_level,
        "verification_coverage": rep.verification_coverage,
        "consistency_score": rep.consistency_score,
        "document_summary": json.loads(rep.document_summary) if rep.document_summary else {},
        "validation_summary": json.loads(rep.validation_summary) if rep.validation_summary else {},
        "cross_document_summary": json.loads(rep.cross_document_summary) if rep.cross_document_summary else {},
        "risk_summary": json.loads(rep.risk_summary) if rep.risk_summary else {},
        "key_findings": json.loads(rep.key_findings) if rep.key_findings else [],
        "recommendations": json.loads(rep.recommendations) if rep.recommendations else [],
        "executive_summary": rep.executive_summary or "",
        "review_required": rep.review_required,
        "generated_by": rep.generated_by,
        "generation_status": rep.generation_status,
        "processing_time_ms": rep.processing_time_ms
    }


# =============================================================================
# 11. PROCESSING RUNS REPOSITORY (Audit Trail)
# =============================================================================

def create_processing_run(db: Session, application_id: str) -> ProcessingRunModel:
    """Creates a new execution processing run audit record."""
    run = ProcessingRunModel(
        application_id=application_id,
        started_at=datetime.utcnow(),
        status="STARTED",
        current_agent="agent_1",
        total_processing_time_ms=0.0
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def update_processing_run(
    db: Session,
    run_id: int,
    status: str,
    current_agent: str,
    error_message: Optional[str] = None,
    total_time_ms: float = 0.0
) -> Optional[ProcessingRunModel]:
    """Updates a processing run audit record."""
    run = db.query(ProcessingRunModel).filter(ProcessingRunModel.id == run_id).first()
    if run:
        run.status = status
        run.current_agent = current_agent
        if status in ["COMPLETED", "FAILED"]:
            run.completed_at = datetime.utcnow()
        if error_message:
            run.error_message = error_message
        run.total_processing_time_ms = total_time_ms
        db.commit()
        db.refresh(run)
    return run


# =============================================================================
# 12. LOAN POLICY REPOSITORY
# =============================================================================

def upsert_loan_policy(
    db: Session,
    policy_data: Dict[str, Any],
    rules_data: List[Dict[str, Any]],
    required_docs_data: List[Dict[str, Any]],
    required_fields_data: Optional[List[Dict[str, Any]]] = None
) -> LoanPolicyModel:
    """Upserts a loan policy along with its rules and document requirements."""
    policy_id = policy_data["policy_id"]
    existing = db.query(LoanPolicyModel).filter(LoanPolicyModel.policy_id == policy_id).first()
    
    if existing:
        for k, v in policy_data.items():
            setattr(existing, k, v)
        existing.updated_at = datetime.utcnow()
        policy = existing
        # Clear old associated rules/docs/fields to refresh cleanly
        db.query(PolicyRuleModel).filter(PolicyRuleModel.policy_id == policy_id).delete()
        db.query(PolicyRequiredDocumentModel).filter(PolicyRequiredDocumentModel.policy_id == policy_id).delete()
        db.query(PolicyRequiredFieldModel).filter(PolicyRequiredFieldModel.policy_id == policy_id).delete()
    else:
        policy = LoanPolicyModel(**policy_data)
        db.add(policy)

    db.flush()

    for r in rules_data:
        r_copy = dict(r)
        r_copy["policy_id"] = policy_id
        db.add(PolicyRuleModel(**r_copy))

    for d in required_docs_data:
        d_copy = dict(d)
        d_copy["policy_id"] = policy_id
        db.add(PolicyRequiredDocumentModel(**d_copy))

    if required_fields_data:
        for f in required_fields_data:
            f_copy = dict(f)
            f_copy["policy_id"] = policy_id
            db.add(PolicyRequiredFieldModel(**f_copy))

    db.commit()
    db.refresh(policy)
    return policy


def get_policy_by_type(db: Session, loan_type: str) -> Optional[LoanPolicyModel]:
    """Retrieves active loan policy by normalized loan type."""
    normalized = loan_type.strip().lower()
    policies = db.query(LoanPolicyModel).filter(LoanPolicyModel.status == "ACTIVE").all()
    for p in policies:
        if p.loan_type.strip().lower() == normalized or p.loan_type.strip().lower().replace(" ", "_") == normalized.replace(" ", "_"):
            return p
    return None


def list_all_policies(db: Session) -> List[Dict[str, Any]]:
    """Lists all loan policies with rules summary."""
    policies = db.query(LoanPolicyModel).all()
    res = []
    for p in policies:
        res.append({
            "policy_id": p.policy_id,
            "loan_type": p.loan_type,
            "policy_name": p.policy_name,
            "version": p.version,
            "description": p.description,
            "effective_date": p.effective_date,
            "source_type": p.source_type,
            "source_document": p.source_document,
            "source_section": p.source_section,
            "source_page": p.source_page,
            "status": p.status,
            "rules_count": len(p.rules),
            "required_docs_count": len(p.required_documents),
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "rule_code": r.rule_code,
                    "category": r.category,
                    "field_name": r.field_name,
                    "operator": r.operator,
                    "expected_value": r.expected_value,
                    "threshold_value": r.threshold_value,
                    "severity": r.severity,
                    "mandatory": r.mandatory,
                    "error_message": r.error_message,
                    "source_type": r.source_type,
                    "source_document": r.source_document,
                    "source_section": r.source_section,
                    "source_page": r.source_page
                } for r in p.rules
            ],
            "required_documents": [
                {
                    "slot_id": d.slot_id,
                    "document_type": d.document_type,
                    "display_name": d.display_name,
                    "required": d.required
                } for d in p.required_documents
            ]
        })
    return res


# =============================================================================
# 13. FIELD EVIDENCE REPOSITORY
# =============================================================================

def save_field_evidence_batch(
    db: Session,
    evidence_list: List[Dict[str, Any]]
) -> List[FieldEvidenceModel]:
    """Saves a batch of field-level evidence citations."""
    records = []
    for item in evidence_list:
        ev_id = item.get("evidence_id")
        existing = None
        if ev_id:
            existing = db.query(FieldEvidenceModel).filter(FieldEvidenceModel.evidence_id == ev_id).first()
        
        if existing:
            for k, v in item.items():
                setattr(existing, k, v)
            records.append(existing)
        else:
            rec = FieldEvidenceModel(
                evidence_id=item["evidence_id"],
                application_id=item["application_id"],
                document_id=item.get("document_id"),
                slot_id=item.get("slot_id"),
                document_type=item.get("document_type"),
                field_name=item["field_name"],
                extracted_value=item.get("extracted_value"),
                raw_value=item.get("raw_value"),
                normalized_value=item.get("normalized_value"),
                source_page=item.get("source_page", 1),
                snippet=item.get("snippet"),
                bounding_box_json=json.dumps(item["bounding_box"]) if isinstance(item.get("bounding_box"), (dict, list)) else item.get("bounding_box_json"),
                extraction_method=item.get("extraction_method", "Regex / Key-Value Extractor"),
                ocr_used=item.get("ocr_used", False),
                confidence=item.get("confidence", 1.0),
                validation_status=item.get("validation_status", "VALIDATED"),
                created_at=datetime.utcnow()
            )
            db.add(rec)
            records.append(rec)
    db.commit()
    return records


def get_field_evidence_by_application(db: Session, application_id: str) -> List[Dict[str, Any]]:
    """Retrieves all field evidence citations for an application."""
    records = db.query(FieldEvidenceModel).filter(FieldEvidenceModel.application_id == application_id).all()
    res = []
    for r in records:
        res.append({
            "evidence_id": r.evidence_id,
            "application_id": r.application_id,
            "document_id": r.document_id,
            "slot_id": r.slot_id,
            "document_type": r.document_type,
            "field_name": r.field_name,
            "extracted_value": r.extracted_value,
            "raw_value": r.raw_value,
            "normalized_value": r.normalized_value,
            "source_page": r.source_page,
            "snippet": r.snippet,
            "bounding_box": json.loads(r.bounding_box_json) if r.bounding_box_json else None,
            "extraction_method": r.extraction_method,
            "ocr_used": r.ocr_used,
            "confidence": r.confidence,
            "validation_status": r.validation_status,
            "created_at": r.created_at.isoformat() if r.created_at else None
        })
    return res


def get_evidence_by_id(db: Session, evidence_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves a single field evidence by evidence_id."""
    r = db.query(FieldEvidenceModel).filter(FieldEvidenceModel.evidence_id == evidence_id).first()
    if not r:
        return None
    return {
        "evidence_id": r.evidence_id,
        "application_id": r.application_id,
        "document_id": r.document_id,
        "slot_id": r.slot_id,
        "document_type": r.document_type,
        "field_name": r.field_name,
        "extracted_value": r.extracted_value,
        "source_page": r.source_page,
        "snippet": r.snippet,
        "extraction_method": r.extraction_method,
        "ocr_used": r.ocr_used,
        "confidence": r.confidence,
        "validation_status": r.validation_status
    }


# =============================================================================
# 14. DECISION GRAPH REPOSITORY
# =============================================================================

def save_decision_graph(
    db: Session,
    graph_data: Dict[str, Any],
    nodes_data: List[Dict[str, Any]],
    edges_data: List[Dict[str, Any]]
) -> DecisionGraphModel:
    """Saves decision graph with all its nodes and edges."""
    graph_id = graph_data["graph_id"]
    existing = db.query(DecisionGraphModel).filter(DecisionGraphModel.graph_id == graph_id).first()
    if existing:
        db.query(DecisionGraphNodeModel).filter(DecisionGraphNodeModel.graph_id == graph_id).delete()
        db.query(DecisionGraphEdgeModel).filter(DecisionGraphEdgeModel.graph_id == graph_id).delete()
        for k, v in graph_data.items():
            setattr(existing, k, v)
        graph = existing
    else:
        graph = DecisionGraphModel(**graph_data)
        db.add(graph)

    db.flush()

    for n in nodes_data:
        n_copy = dict(n)
        n_copy["graph_id"] = graph_id
        if "node_metadata" in n_copy:
            n_copy["node_metadata_json"] = json.dumps(n_copy.pop("node_metadata"))
        db.add(DecisionGraphNodeModel(**n_copy))

    for e in edges_data:
        e_copy = dict(e)
        e_copy["graph_id"] = graph_id
        db.add(DecisionGraphEdgeModel(**e_copy))

    db.commit()
    db.refresh(graph)
    return graph


def get_decision_graph_by_application(db: Session, application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves decision graph with nodes and edges for an application."""
    graph = (
        db.query(DecisionGraphModel)
        .filter(DecisionGraphModel.application_id == application_id)
        .order_by(DecisionGraphModel.created_at.desc())
        .first()
    )
    if not graph:
        return None

    return {
        "graph_id": graph.graph_id,
        "application_id": graph.application_id,
        "total_nodes": graph.total_nodes,
        "total_edges": graph.total_edges,
        "passed_nodes": graph.passed_nodes,
        "failed_nodes": graph.failed_nodes,
        "warning_nodes": graph.warning_nodes,
        "info_nodes": graph.info_nodes,
        "created_at": graph.created_at.isoformat() if graph.created_at else None,
        "nodes": [
            {
                "node_id": n.node_id,
                "stage_name": n.stage_name,
                "label": n.label,
                "status": n.status,
                "agent_name": n.agent_name,
                "description": n.description,
                "evidence_count": n.evidence_count,
                "citation_count": n.citation_count,
                "node_metadata": json.loads(n.node_metadata_json) if n.node_metadata_json else {}
            } for n in graph.nodes
        ],
        "edges": [
            {
                "source": e.source_node_id,
                "target": e.target_node_id,
                "label": e.edge_label,
                "type": e.edge_type
            } for e in graph.edges
        ]
    }


# =============================================================================
# 15. ELIGIBILITY RESULTS REPOSITORY
# =============================================================================

def save_eligibility_result(
    db: Session,
    eligibility_data: Dict[str, Any],
    rule_results_data: List[Dict[str, Any]]
) -> EligibilityResultModel:
    """Saves eligibility engine decision and rule results."""
    el_id = eligibility_data["eligibility_id"]
    existing = db.query(EligibilityResultModel).filter(EligibilityResultModel.eligibility_id == el_id).first()
    if existing:
        db.query(EligibilityRuleResultModel).filter(EligibilityRuleResultModel.eligibility_id == el_id).delete()
        for k, v in eligibility_data.items():
            if k == "reasons" and isinstance(v, list):
                setattr(existing, "reasons_json", json.dumps(v))
            else:
                setattr(existing, k, v)
        res = existing
    else:
        d_copy = dict(eligibility_data)
        if "reasons" in d_copy and isinstance(d_copy["reasons"], list):
            d_copy["reasons_json"] = json.dumps(d_copy.pop("reasons"))
        res = EligibilityResultModel(**d_copy)
        db.add(res)

    db.flush()

    for rr in rule_results_data:
        rr_copy = dict(rr)
        rr_copy["eligibility_id"] = el_id
        db.add(EligibilityRuleResultModel(**rr_copy))

    db.commit()
    db.refresh(res)
    return res


def get_eligibility_by_application(db: Session, application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves eligibility engine decision and rule breakdown for an application."""
    res = (
        db.query(EligibilityResultModel)
        .filter(EligibilityResultModel.application_id == application_id)
        .order_by(EligibilityResultModel.created_at.desc())
        .first()
    )
    if not res:
        return None

    return {
        "eligibility_id": res.eligibility_id,
        "application_id": res.application_id,
        "loan_type": res.loan_type,
        "decision": res.decision,
        "confidence": res.confidence,
        "reasons": json.loads(res.reasons_json) if res.reasons_json else [],
        "rules_evaluated_count": res.rules_evaluated_count,
        "rules_passed_count": res.rules_passed_count,
        "rules_failed_count": res.rules_failed_count,
        "rules_skipped_count": res.rules_skipped_count,
        "processing_time_ms": res.processing_time_ms,
        "created_at": res.created_at.isoformat() if res.created_at else None,
        "rule_results": [
            {
                "rule_code": rr.rule_code,
                "category": rr.category,
                "field_name": rr.field_name,
                "operator": rr.operator,
                "expected_value": rr.expected_value,
                "actual_value": rr.actual_value,
                "status": rr.status,
                "mandatory": rr.mandatory,
                "severity": rr.severity,
                "failure_reason": rr.failure_reason,
                "evidence_id": rr.evidence_id,
                "policy_rule_id": rr.policy_rule_id
            } for rr in res.rule_results
        ]
    }


# =============================================================================
# 16. AGENT EXECUTION LOGS & TELEMETRY REPOSITORY
# =============================================================================

def log_agent_execution(
    db: Session,
    application_id: str,
    agent_name: str,
    started_at: datetime,
    completed_at: datetime,
    duration_ms: float,
    status: str = "SUCCESS",
    input_summary: Optional[Dict[str, Any]] = None,
    output_summary: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> AgentExecutionLogModel:
    """Logs individual agent execution telemetry."""
    log_rec = AgentExecutionLogModel(
        application_id=application_id,
        agent_name=agent_name,
        started_at=started_at,
        completed_at=completed_at,
        duration_ms=duration_ms,
        status=status,
        input_summary_json=json.dumps(input_summary) if input_summary else None,
        output_summary_json=json.dumps(output_summary) if output_summary else None,
        error_message=error_message,
        created_at=datetime.utcnow()
    )
    db.add(log_rec)
    db.commit()
    db.refresh(log_rec)
    return log_rec


def save_telemetry_metrics(
    db: Session,
    application_id: str,
    metrics: Dict[str, Any]
) -> TelemetryMetricsModel:
    """Saves end-to-end telemetry metrics for an application run."""
    agent_durations = metrics.get("agent_durations", {})
    rec = TelemetryMetricsModel(
        application_id=application_id,
        total_pipeline_duration_ms=metrics.get("total_pipeline_duration_ms", 0.0),
        agent_durations_json=json.dumps(agent_durations),
        ocr_duration_ms=metrics.get("ocr_duration_ms", 0.0),
        validation_duration_ms=metrics.get("validation_duration_ms", 0.0),
        cross_doc_duration_ms=metrics.get("cross_doc_duration_ms", 0.0),
        risk_duration_ms=metrics.get("risk_duration_ms", 0.0),
        eligibility_duration_ms=metrics.get("eligibility_duration_ms", 0.0),
        decision_graph_duration_ms=metrics.get("decision_graph_duration_ms", 0.0),
        report_duration_ms=metrics.get("report_duration_ms", 0.0),
        documents_count=metrics.get("documents_count", 0),
        fields_count=metrics.get("fields_count", 0),
        citations_count=metrics.get("citations_count", 0),
        memory_mb=metrics.get("memory_mb", 0.0),
        created_at=datetime.utcnow()
    )
    db.add(rec)
    db.commit()
    db.refresh(rec)
    return rec


def get_telemetry_by_application(db: Session, application_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves telemetry summary and agent execution breakdown for an application."""
    metric = (
        db.query(TelemetryMetricsModel)
        .filter(TelemetryMetricsModel.application_id == application_id)
        .order_by(TelemetryMetricsModel.created_at.desc())
        .first()
    )
    logs = (
        db.query(AgentExecutionLogModel)
        .filter(AgentExecutionLogModel.application_id == application_id)
        .order_by(AgentExecutionLogModel.started_at.asc())
        .all()
    )

    if not metric and not logs:
        return None

    agent_durations = json.loads(metric.agent_durations_json) if metric and metric.agent_durations_json else {}
    return {
        "application_id": application_id,
        "total_pipeline_duration_ms": metric.total_pipeline_duration_ms if metric else sum(l.duration_ms for l in logs),
        "agent_durations": agent_durations,
        "ocr_duration_ms": metric.ocr_duration_ms if metric else 0.0,
        "validation_duration_ms": metric.validation_duration_ms if metric else 0.0,
        "cross_doc_duration_ms": metric.cross_doc_duration_ms if metric else 0.0,
        "risk_duration_ms": metric.risk_duration_ms if metric else 0.0,
        "eligibility_duration_ms": metric.eligibility_duration_ms if metric else 0.0,
        "decision_graph_duration_ms": metric.decision_graph_duration_ms if metric else 0.0,
        "report_duration_ms": metric.report_duration_ms if metric else 0.0,
        "documents_count": metric.documents_count if metric else 0,
        "fields_count": metric.fields_count if metric else 0,
        "citations_count": metric.citations_count if metric else 0,
        "memory_mb": metric.memory_mb if metric else 0.0,
        "created_at": metric.created_at.isoformat() if metric and metric.created_at else None,
        "agent_logs": [
            {
                "agent_name": l.agent_name,
                "started_at": l.started_at.isoformat() if l.started_at else None,
                "completed_at": l.completed_at.isoformat() if l.completed_at else None,
                "duration_ms": l.duration_ms,
                "status": l.status,
                "input_summary": json.loads(l.input_summary_json) if l.input_summary_json else None,
                "output_summary": json.loads(l.output_summary_json) if l.output_summary_json else None,
                "error_message": l.error_message
            } for l in logs
        ]
    }


# =============================================================================
# 13. USERS REPOSITORY
# =============================================================================

def create_user(
    db: Session,
    email: str,
    full_name: str,
    password_hash: str,
    role: str,
    branch_id: Optional[str] = None
) -> User:
    """Creates a new user account."""
    user = User(
        email=email.lower().strip(),
        full_name=full_name,
        password_hash=password_hash,
        role=role,
        branch_id=branch_id,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Retrieves a user by email address."""
    return db.query(User).filter(User.email == email.lower().strip()).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Retrieves a user by primary ID."""
    return db.query(User).filter(User.id == user_id).first()


def list_users(db: Session, role: Optional[str] = None, branch_id: Optional[str] = None) -> List[User]:
    """Lists users with optional role and branch filtering."""
    q = db.query(User)
    if role:
        q = q.filter(User.role == role)
    if branch_id:
        q = q.filter(User.branch_id == branch_id)
    return q.order_by(User.created_at.desc()).all()


# =============================================================================
# 14. MANAGER ANALYTICS & MONITORING REPOSITORY
# =============================================================================

def _parse_filter_date(d_val: Any) -> Optional[datetime]:
    """Parses date string or returns datetime."""
    if not d_val:
        return None
    if isinstance(d_val, datetime):
        return d_val
    try:
        from dateutil import parser
        return parser.parse(str(d_val))
    except Exception:
        try:
            return datetime.fromisoformat(str(d_val).replace("Z", "+00:00").split("+")[0])
        except Exception:
            return None


def _filter_applications_query(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None,
    loan_type: Optional[str] = None,
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    employee_id: Optional[str] = None,
    search: Optional[str] = None
):
    """Internal helper to apply common filters to LoanApplication query."""
    q = db.query(LoanApplication)
    
    dt_from = _parse_filter_date(from_date)
    if dt_from:
        q = q.filter(LoanApplication.created_at >= dt_from)
        
    dt_to = _parse_filter_date(to_date)
    if dt_to:
        q = q.filter(LoanApplication.created_at <= dt_to)

    if branch_id:
        q = q.filter(LoanApplication.branch_id == branch_id)
    if loan_type:
        q = q.filter(LoanApplication.loan_type == loan_type)
    if status:
        q = q.filter(LoanApplication.status == status)
    if risk_level:
        q = q.filter(LoanApplication.risk_level == risk_level)
    if employee_id:
        q = q.filter(LoanApplication.employee_id == employee_id)
    if search:
        search_pattern = f"%{search}%"
        q = q.filter(
            or_(
                LoanApplication.application_id.ilike(search_pattern),
                LoanApplication.applicant_name.ilike(search_pattern)
            )
        )
    return q


def get_analytics_summary(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None,
    loan_type: Optional[str] = None
) -> Dict[str, Any]:
    """Feature A: Summary metric card statistics."""
    q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id, loan_type=loan_type)
    apps = q.all()

    total = len(apps)
    approved = 0
    rejected = 0
    human_review = 0
    insufficient = 0

    for a in apps:
        st = (a.status or "").upper()
        if st in ["APPROVED", "PASS", "ELIGIBLE"]:
            approved += 1
        elif st in ["REJECTED", "FAIL", "NOT_ELIGIBLE"]:
            rejected += 1
        elif st in ["HUMAN_REVIEW", "HUMAN_REVIEW_REQUIRED"]:
            human_review += 1
        elif st in ["INSUFFICIENT_EVIDENCE", "INCOMPLETE", "NOT_STARTED"]:
            insufficient += 1
        elif st == "COMPLETED":
            # Check FinalReport or Risk
            fr = db.query(FinalReportModel).filter(FinalReportModel.application_id == a.application_id).first()
            if fr and fr.decision:
                fdec = fr.decision.upper()
                if fdec in ["APPROVED", "PASS", "ELIGIBLE"]:
                    approved += 1
                elif fdec in ["REJECTED", "FAIL", "NOT_ELIGIBLE"]:
                    rejected += 1
                elif fdec in ["HUMAN_REVIEW", "HUMAN_REVIEW_REQUIRED"]:
                    human_review += 1
                else:
                    insufficient += 1
            else:
                approved += 1

    durations = [a.processing_time for a in apps if a.processing_time and a.processing_time > 0]
    avg_dur = round(sum(durations) / len(durations), 2) if durations else 450.0

    return {
        "total_applications": total,
        "approved_loans": approved,
        "approved_applications": approved,
        "rejected_loans": rejected,
        "rejected_applications": rejected,
        "human_review_required": human_review,
        "human_review_applications": human_review,
        "insufficient_evidence": insufficient,
        "insufficient_evidence_applications": insufficient,
        "approval_rate": round(approved / total * 100, 1) if total else 0.0,
        "rejection_rate": round(rejected / total * 100, 1) if total else 0.0,
        "human_review_rate": round(human_review / total * 100, 1) if total else 0.0,
        "avg_processing_time_ms": avg_dur
    }


def get_loan_type_statistics(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Feature B: Loan-type-wise counts covering all 10 canonical loan types."""
    from shared.policy import LOAN_DOCUMENT_POLICY, LOAN_TYPE_NAMES

    q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id)
    rows = q.with_entities(LoanApplication.loan_type, func.count(LoanApplication.id)).group_by(LoanApplication.loan_type).all()

    counts_map = {}
    for lt, cnt in rows:
        norm_lt = (lt or "").lower().replace(" ", "_").replace("-", "_")
        counts_map[norm_lt] = counts_map.get(norm_lt, 0) + cnt

    result = []
    for lt_key in LOAN_DOCUMENT_POLICY.keys():
        disp_name = LOAN_TYPE_NAMES.get(lt_key, lt_key.replace("_", " ").title())
        result.append({
            "loan_type": lt_key,
            "display_name": disp_name,
            "count": counts_map.get(lt_key, 0)
        })

    return result


def get_monthly_trends(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None,
    loan_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Feature C: Monthly application volume and approval/rejection trends."""
    q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id, loan_type=loan_type)
    apps = q.order_by(asc(LoanApplication.created_at)).all()

    months_data: Dict[str, Dict[str, int]] = {}
    for a in apps:
        m_str = a.created_at.strftime("%Y-%m") if a.created_at else "Unknown"
        if m_str not in months_data:
            months_data[m_str] = {"total": 0, "approved": 0, "rejected": 0}

        months_data[m_str]["total"] += 1
        st = (a.status or "").upper()
        if st in ["APPROVED", "PASS", "ELIGIBLE", "COMPLETED"]:
            months_data[m_str]["approved"] += 1
        elif st in ["REJECTED", "FAIL", "NOT_ELIGIBLE"]:
            months_data[m_str]["rejected"] += 1

    return [
        {
            "month": m,
            "total": d["total"],
            "approved": d["approved"],
            "rejected": d["rejected"],
            "approval_rate": round((d["approved"] / d["total"]) * 100, 1) if d["total"] else 0.0
        } for m, d in sorted(months_data.items())
    ]


def get_risk_distribution(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None,
    loan_type: Optional[str] = None
) -> Dict[str, Any]:
    """Feature D: Risk distribution counts and breakdown."""
    q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id, loan_type=loan_type)
    apps = q.all()

    low = 0
    medium = 0
    high = 0
    unknown = 0

    for a in apps:
        rl = a.risk_level
        if not rl:
            # Check risk assessment table
            ra = db.query(RiskAssessmentModel).filter(RiskAssessmentModel.application_id == a.application_id).first()
            if ra and ra.risk_level:
                rl = ra.risk_level

        if rl:
            rl_upper = rl.upper()
            if rl_upper in ["LOW", "MINIMAL"]:
                low += 1
            elif rl_upper in ["MEDIUM", "MODERATE"]:
                medium += 1
            elif rl_upper in ["HIGH", "CRITICAL"]:
                high += 1
            else:
                unknown += 1
        else:
            unknown += 1

    total = len(apps)
    return {
        "low": low,
        "medium": medium,
        "high": high,
        "unknown": unknown,
        "total": total,
        "distribution": [
            {"level": "Low Risk", "risk_level": "LOW", "count": low, "percentage": round(low / total * 100, 1) if total else 0.0},
            {"level": "Medium Risk", "risk_level": "MEDIUM", "count": medium, "percentage": round(medium / total * 100, 1) if total else 0.0},
            {"level": "High Risk", "risk_level": "HIGH", "count": high, "percentage": round(high / total * 100, 1) if total else 0.0},
            {"level": "Unknown / Pending", "risk_level": "UNKNOWN", "count": unknown, "percentage": round(unknown / total * 100, 1) if total else 0.0},
        ]
    }


def get_processing_performance(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None
) -> Dict[str, Any]:
    """Feature E: Average application processing time and throughput metrics."""
    q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id)
    apps = q.all()

    durations: List[float] = []
    total_processed = 0
    currently_processing = 0
    failed_processing = 0

    for a in apps:
        st = (a.status or "").upper()
        if st in ["PROCESSING", "IN_PROGRESS"]:
            currently_processing += 1
        elif st in ["FAILED", "ERROR"]:
            failed_processing += 1
        elif st in ["COMPLETED", "APPROVED", "REJECTED", "HUMAN_REVIEW", "INSUFFICIENT_EVIDENCE"]:
            total_processed += 1
            if a.processing_time and a.processing_time > 0:
                durations.append(a.processing_time)
            elif a.completed_at and a.created_at and a.completed_at > a.created_at:
                durations.append((a.completed_at - a.created_at).total_seconds() * 1000.0)

    # If no durations in loan_applications, check telemetry_metrics table
    if not durations:
        tm_rows = db.query(TelemetryMetricsModel.total_pipeline_duration_ms).all()
        for r in tm_rows:
            if r[0] and r[0] > 0:
                durations.append(r[0])
                if total_processed == 0:
                    total_processed += 1

    avg_time_ms = round(sum(durations) / len(durations), 2) if durations else None
    avg_time_sec = round(avg_time_ms / 1000.0, 2) if avg_time_ms is not None else None

    return {
        "average_processing_time_ms": avg_time_ms,
        "average_processing_time_seconds": avg_time_sec,
        "average_processing_time": f"{avg_time_sec}s" if avg_time_sec is not None else "N/A",
        "total_processed": total_processed,
        "currently_processing": currently_processing,
        "failed_processing": failed_processing
    }


def get_agent_performance(
    db: Session,
    from_date: Any = None,
    to_date: Any = None
) -> List[Dict[str, Any]]:
    """Feature F: Execution counts, success rates, and average runtime for Agents 1-6."""
    canonical_agents = [
        ("agent_1", "Agent 1 - Document Classification"),
        ("agent_2", "Agent 2 - Information Extraction"),
        ("agent_3", "Agent 3 - Validation"),
        ("agent_4", "Agent 4 - Cross-Document Verification"),
        ("agent_5", "Agent 5 - Risk Analysis"),
        ("agent_6", "Agent 6 - Final Report")
    ]

    q = db.query(AgentExecutionLogModel)
    dt_from = _parse_filter_date(from_date)
    if dt_from:
        q = q.filter(AgentExecutionLogModel.created_at >= dt_from)
    dt_to = _parse_filter_date(to_date)
    if dt_to:
        q = q.filter(AgentExecutionLogModel.created_at <= dt_to)

    logs = q.all()
    logs_by_agent: Dict[str, List[AgentExecutionLogModel]] = {}
    for l in logs:
        key = l.agent_name.lower().strip()
        logs_by_agent.setdefault(key, []).append(l)

    result = []
    for agent_id, agent_title in canonical_agents:
        agent_logs = logs_by_agent.get(agent_id, [])
        total_exec = len(agent_logs)
        succ = sum(1 for x in agent_logs if (x.status or "").upper() == "SUCCESS")
        fail = sum(1 for x in agent_logs if (x.status or "").upper() != "SUCCESS")
        avg_dur = round(sum(x.duration_ms for x in agent_logs) / total_exec, 2) if total_exec else 0.0

        succ_rate = round((succ / total_exec) * 100, 1) if total_exec else 100.0
        result.append({
            "agent_id": agent_id,
            "agent_name": agent_title,
            "total_executions": total_exec,
            "successful_executions": succ,
            "failed_executions": fail,
            "success_rate": succ_rate,
            "avg_duration_ms": avg_dur,
            "average_execution_time_ms": avg_dur
        })

    return result


def get_validation_analytics(
    db: Session,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None,
    loan_type: Optional[str] = None
) -> Dict[str, Any]:
    """Feature G: Validation failure aggregates, slot discrepancies, and error frequencies."""
    val_q = db.query(ValidationResultModel)
    doc_q = db.query(DocumentModel)

    dt_from = _parse_filter_date(from_date)
    if dt_from:
        val_q = val_q.filter(ValidationResultModel.created_at >= dt_from)
        doc_q = doc_q.filter(DocumentModel.uploaded_at >= dt_from)

    dt_to = _parse_filter_date(to_date)
    if dt_to:
        val_q = val_q.filter(ValidationResultModel.created_at <= dt_to)
        doc_q = doc_q.filter(DocumentModel.uploaded_at <= dt_to)

    val_records = val_q.all()
    doc_records = doc_q.all()

    total_failures = 0
    invalid_fields = 0
    error_reasons: Dict[str, int] = {}

    for vr in val_records:
        st = (vr.status or "").upper()
        if st == "FAIL":
            total_failures += 1
            if vr.field_name:
                invalid_fields += 1
            msg = vr.explanation or vr.field_name or "Validation rule failure"
            error_reasons[msg] = error_reasons.get(msg, 0) + 1

    wrong_docs = sum(1 for d in doc_records if d.upload_status == "rejected")
    rejected_types: Dict[str, int] = {}
    for d in doc_records:
        if d.upload_status == "rejected" and d.document_type:
            rejected_types[d.document_type] = rejected_types.get(d.document_type, 0) + 1

    # Missing documents count across applications
    app_q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id, loan_type=loan_type)
    missing_docs = 0
    for a in app_q.all():
        if a.status in ["INCOMPLETE", "NOT_STARTED"]:
            missing_docs += 1

    top_errors = sorted([{"error": k, "count": v} for k, v in error_reasons.items()], key=lambda x: x["count"], reverse=True)[:5]
    top_rejected = sorted([{"document_type": k, "count": v} for k, v in rejected_types.items()], key=lambda x: x["count"], reverse=True)[:5]
    top_reasons_str_list = [e["error"] for e in top_errors]

    return {
        "total_validation_failures": total_failures,
        "missing_document_count": missing_docs,
        "missing_documents_count": missing_docs,
        "wrong_document_count": wrong_docs,
        "invalid_field_count": invalid_fields,
        "invalid_fields_count": invalid_fields,
        "top_rejection_reasons": top_reasons_str_list,
        "most_common_validation_errors": top_errors,
        "frequently_rejected_document_types": top_rejected
    }


def get_high_risk_applications(
    db: Session,
    limit: int = 10,
    from_date: Any = None,
    to_date: Any = None,
    branch_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Feature D (table): High-risk applications with reasons and status."""
    q = _filter_applications_query(db, from_date=from_date, to_date=to_date, branch_id=branch_id)
    q = q.filter(LoanApplication.risk_level.in_(["HIGH", "CRITICAL"]))
    apps = q.order_by(desc(LoanApplication.created_at)).limit(limit).all()

    # Also check risk assessments if apps list is empty
    if not apps:
        ra_high = db.query(RiskAssessmentModel).filter(RiskAssessmentModel.risk_level.in_(["HIGH", "CRITICAL"])).limit(limit).all()
        app_ids = [ra.application_id for ra in ra_high]
        if app_ids:
            apps = db.query(LoanApplication).filter(LoanApplication.application_id.in_(app_ids)).limit(limit).all()

    result = []
    for a in apps:
        reasons = []
        ra = db.query(RiskAssessmentModel).filter(RiskAssessmentModel.application_id == a.application_id).first()
        if ra:
            if ra.overall_reason:
                reasons.append(ra.overall_reason)
            rf_list = db.query(RiskFactorModel).filter(RiskFactorModel.risk_assessment_id == ra.id).all()
            for rf in rf_list:
                if rf.description:
                    reasons.append(rf.description)

        reason_text = "; ".join(reasons[:2]) if reasons else "High cumulative risk score flagged by Agent 5"
        result.append({
            "application_id": a.application_id,
            "applicant_name": a.applicant_name or "Applicant",
            "loan_type": a.loan_type,
            "risk_level": a.risk_level or (ra.risk_level if ra else "HIGH"),
            "risk_reasons": reason_text,
            "application_status": a.status,
            "created_at": a.created_at.isoformat() if a.created_at else None
        })

    return result


def get_monitored_applications(
    db: Session,
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None,
    loan_type: Optional[str] = None,
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    employee_id: Optional[str] = None,
    branch_id: Optional[str] = None,
    from_date: Any = None,
    to_date: Any = None,
    sort_by: str = "created_at",
    sort_order: str = "desc"
) -> Dict[str, Any]:
    """Feature H: Searchable, filtered, sorted, paginated applications monitoring table."""
    q = _filter_applications_query(
        db,
        from_date=from_date,
        to_date=to_date,
        branch_id=branch_id,
        loan_type=loan_type,
        status=status,
        risk_level=risk_level,
        employee_id=employee_id,
        search=search
    )

    total_count = q.count()

    # Apply sorting
    sort_col = getattr(LoanApplication, sort_by, LoanApplication.created_at)
    if sort_order.lower() == "asc":
        q = q.order_by(asc(sort_col))
    else:
        q = q.order_by(desc(sort_col))

    # Apply pagination
    page = max(1, page)
    page_size = max(1, min(page_size, 100))
    offset = (page - 1) * page_size
    apps = q.offset(offset).limit(page_size).all()

    import math
    total_pages = math.ceil(total_count / page_size) if total_count else 1

    items = []
    for a in apps:
        proc_time_str = f"{round(a.processing_time / 1000.0, 2)}s" if a.processing_time else ("N/A" if a.status != "COMPLETED" else "0.5s")
        items.append({
            "application_id": a.application_id,
            "applicant_name": a.applicant_name or "Applicant",
            "loan_type": a.loan_type,
            "employee_id": a.employee_id or "Unassigned",
            "branch_id": a.branch_id or "BR-MUMBAI-01",
            "status": a.status,
            "risk_level": a.risk_level or "LOW",
            "created_at": a.created_at.isoformat() if a.created_at else None,
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
            "processing_time": proc_time_str
        })

    return {
        "items": items,
        "total": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages
    }


