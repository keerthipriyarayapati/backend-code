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
    ProcessingRunModel
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
    status: str = "NOT_STARTED"
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
                src_text = src.get("snippet") or src.get("raw_text")
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
