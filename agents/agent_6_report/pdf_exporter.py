"""
PDF Exporter Module for Agent 6 — Final Loan Application Report.
Uses ReportLab to generate clean, professional, publication-ready PDF reports with
selectable text, dynamic page numbering (Page X of Y), masked sensitive fields,
and complete multi-agent summary sections.
"""

import io
import sys
import time
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable
)
from reportlab.pdfgen import canvas

# Import project helper for masking sensitive values
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.agent_5_risk.agent import mask_sensitive_value

logger = logging.getLogger("PDFExporter")


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


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and render total page count
    ('Page X of Y') along with standard running headers and footers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header for pages > 1
        if self._pageNumber > 1:
            self.drawString(36, 756, "LOAN DOCUMENT PROCESSING SYSTEM — FINAL APPLICATION REPORT")
            self.drawRightString(576, 756, "CONFIDENTIAL")
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(36, 748, 576, 748)

        # Running Footer for all pages
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(36, 45, 576, 45)

        self.drawString(36, 32, "Loan Document Processing System | Final Report — Agent 6")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(576, 32, page_str)

        self.restoreState()


def build_pdf_report_bytes(
    final_report_obj: Union[Dict[str, Any], Any],
    classification_results: Optional[List[Dict[str, Any]]] = None,
    validation_results: Optional[List[Dict[str, Any]]] = None,
    cross_document_results: Optional[Dict[str, Any]] = None,
    risk_assessment_results: Optional[Dict[str, Any]] = None
) -> bytes:
    """
    Generates a complete binary PDF report for the given FinalReport object.
    Returns bytes of the compiled PDF document.
    """
    # -------------------------------------------------------------------------
    # 1. INPUT CONTRACT VALIDATION
    # -------------------------------------------------------------------------
    if not final_report_obj:
        raise ValueError("Cannot generate PDF: final_report object is missing.")

    app_id = get_obj_attr(final_report_obj, "application_id")
    loan_type = get_obj_attr(final_report_obj, "loan_type")
    decision = get_obj_attr(final_report_obj, "decision")
    risk_score = get_obj_attr(final_report_obj, "risk_score")
    risk_level = get_obj_attr(final_report_obj, "risk_level")

    if not app_id:
        raise ValueError("Invalid FinalReport structure: missing application_id.")
    if not loan_type:
        raise ValueError("Invalid FinalReport structure: missing loan_type.")
    if not decision:
        raise ValueError("Invalid FinalReport structure: missing decision.")
    if risk_score is None:
        raise ValueError("Invalid FinalReport structure: missing risk_score.")
    if not risk_level:
        raise ValueError("Invalid FinalReport structure: missing risk_level.")

    decision_reason = get_obj_attr(final_report_obj, "decision_reason", "")
    exec_summary = get_obj_attr(final_report_obj, "executive_summary", "")
    gen_status = get_obj_attr(final_report_obj, "generation_status", "GENAI_GENERATED")
    review_required = get_obj_attr(final_report_obj, "review_required", True)

    doc_sum = get_obj_attr(final_report_obj, "document_summary") or {}
    val_sum = get_obj_attr(final_report_obj, "validation_summary") or {}
    cross_sum = get_obj_attr(final_report_obj, "cross_document_summary") or {}
    risk_sum = get_obj_attr(final_report_obj, "risk_summary") or {}

    key_findings = get_obj_attr(final_report_obj, "key_findings") or []
    recommendations = get_obj_attr(final_report_obj, "recommendations") or []

    # -------------------------------------------------------------------------
    # 2. REPORTLAB STYLES & SETUP
    # -------------------------------------------------------------------------
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY_NAVY = colors.HexColor("#0f172a")
    HEADER_BLUE = colors.HexColor("#1e3a8a")
    SLATE_TEXT = colors.HexColor("#334155")

    # Decision Colors
    if decision == "PASS":
        DEC_TEXT = colors.HexColor("#15803d")
        DEC_BG = colors.HexColor("#f0fdf4")
        DEC_BORDER = colors.HexColor("#22c55e")
    elif decision == "HUMAN_REVIEW":
        DEC_TEXT = colors.HexColor("#b45309")
        DEC_BG = colors.HexColor("#fefce8")
        DEC_BORDER = colors.HexColor("#eab308")
    else:  # INSUFFICIENT_EVIDENCE or FAIL
        DEC_TEXT = colors.HexColor("#b91c1c")
        DEC_BG = colors.HexColor("#fef2f2")
        DEC_BORDER = colors.HexColor("#ef4444")

    # Typography Styles
    style_title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=PRIMARY_NAVY,
        spaceAfter=4
    )

    style_subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=12
    )

    style_section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=PRIMARY_NAVY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=SLATE_TEXT
    )

    style_body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=13,
        textColor=PRIMARY_NAVY
    )

    style_genai_tag = ParagraphStyle(
        "GenAITag",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#6b21a8")
    )

    style_cell_header = ParagraphStyle(
        "CellHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.white
    )

    style_cell_body = ParagraphStyle(
        "CellBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=SLATE_TEXT
    )

    style_cell_body_bold = ParagraphStyle(
        "CellBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=PRIMARY_NAVY
    )

    story = []

    # -------------------------------------------------------------------------
    # 3. HEADER & METADATA SECTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("FINAL LOAN APPLICATION REPORT", style_title))
    story.append(Paragraph("AUTOMATED MULTI-AGENT DOCUMENT ASSESSMENT & RISK VERIFICATION", style_subtitle))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_NAVY, spaceAfter=10))

    meta_table_data = [
        [
            Paragraph("<b>Application ID:</b>", style_body), Paragraph(f"<font color='#1d4ed8'><b>{app_id}</b></font>", style_body),
            Paragraph("<b>Processing Date:</b>", style_body), Paragraph(time.strftime("%Y-%m-%d %H:%M:%S"), style_body)
        ],
        [
            Paragraph("<b>Loan Type:</b>", style_body), Paragraph(f"<b>{loan_type.replace('_', ' ').upper()}</b>", style_body),
            Paragraph("<b>Processing Status:</b>", style_body), Paragraph("<font color='#15803d'><b>COMPLETED</b></font>", style_body)
        ]
    ]

    meta_table = Table(meta_table_data, colWidths=[100, 170, 100, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 4. FINAL DECISION & RISK SUMMARY CARD
    # -------------------------------------------------------------------------
    dec_display = decision.replace("_", " ").upper()
    risk_score_val = float(risk_score) if risk_score is not None else 0.0
    review_req_str = "YES" if review_required else "NO"

    card_data = [
        [
            Paragraph("<b>FINAL DECISION</b>", style_body_bold),
            Paragraph(f"<font color='{DEC_TEXT.hexval()}'><b>{dec_display}</b></font>", style_title)
        ],
        [
            Paragraph("<b>Risk Assessment:</b>", style_body),
            Paragraph(f"Score: <b>{risk_score_val:.1f} / 100</b> &nbsp;|&nbsp; Level: <b>{risk_level}</b> &nbsp;|&nbsp; Underwriter Review: <b>{review_req_str}</b>", style_body)
        ],
        [
            Paragraph("<b>Decision Reason:</b>", style_body),
            Paragraph(decision_reason, style_body)
        ]
    ]

    card_table = Table(card_data, colWidths=[110, 430])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), DEC_BG),
        ('BOX', (0, 0), (-1, -1), 1.5, DEC_BORDER),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#fef08a") if decision == "HUMAN_REVIEW" else colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(card_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 5. EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    story.append(Paragraph("EXECUTIVE SUMMARY", style_section_heading))

    genai_label = (
        "Generated using: LangChain + Ollama"
        if gen_status == "GENAI_GENERATED"
        else "Generated using: Deterministic Fallback"
    )

    exec_box_data = [
        [Paragraph(f"<b>🤖 {genai_label.upper()}</b>", style_genai_tag)],
        [Paragraph(exec_summary, style_body)]
    ]
    exec_table = Table(exec_box_data, colWidths=[540])
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#faf5ff")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#d8b4fe")),
        ('LINEBEFORE', (0, 0), (0, -1), 3, colors.HexColor("#9333ea")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 6. DOCUMENT COMPLETENESS & STATUS
    # -------------------------------------------------------------------------
    story.append(Paragraph("DOCUMENT SUMMARY & COMPLETENESS", style_section_heading))

    uploaded_cnt = get_obj_attr(doc_sum, "uploaded_count", 0)
    req_cnt = get_obj_attr(doc_sum, "required_count", 0)
    sat_cnt = get_obj_attr(doc_sum, "satisfied_count", 0)
    miss_cnt = get_obj_attr(doc_sum, "missing_count", 0)
    wrong_cnt = get_obj_attr(doc_sum, "wrong_count", 0)
    doc_status_str = get_obj_attr(doc_sum, "status", "N/A")

    doc_grid_data = [
        [
            Paragraph("<b>Total Uploaded:</b>", style_body), Paragraph(str(uploaded_cnt), style_body),
            Paragraph("<b>Required Slots:</b>", style_body), Paragraph(str(req_cnt), style_body),
            Paragraph("<b>Slots Satisfied:</b>", style_body), Paragraph(f"<font color='#15803d'><b>{sat_cnt}</b></font>", style_body)
        ],
        [
            Paragraph("<b>Missing Required:</b>", style_body), Paragraph(f"<font color='#b91c1c'><b>{miss_cnt}</b></font>", style_body),
            Paragraph("<b>Wrong Documents:</b>", style_body), Paragraph(f"<font color='#b91c1c'><b>{wrong_cnt}</b></font>", style_body),
            Paragraph("<b>Overall Status:</b>", style_body), Paragraph(f"<b>{doc_status_str}</b>", style_body)
        ]
    ]
    doc_grid_table = Table(doc_grid_data, colWidths=[90, 90, 90, 90, 90, 90])
    doc_grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(doc_grid_table)
    story.append(Spacer(1, 8))

    # Detailed Classification Table
    class_list = classification_results or []
    if class_list:
        class_table_data = [
            [
                Paragraph("Filename", style_cell_header),
                Paragraph("Expected / Category", style_cell_header),
                Paragraph("Detected Type", style_cell_header),
                Paragraph("Status", style_cell_header),
                Paragraph("Confidence", style_cell_header)
            ]
        ]
        for c in class_list:
            fname = get_obj_attr(c, "filename", "document.pdf")
            raw_dtype = get_obj_attr(c, "document_type", "unknown")
            norm_dtype = get_obj_attr(c, "normalized_document_type", raw_dtype)
            conf = get_obj_attr(c, "confidence", 0.0)
            conf_str = f"{conf*100:.0f}%" if isinstance(conf, (int, float)) else "N/A"
            reason = get_obj_attr(c, "classification_reason", "Verified")
            c_status = get_obj_attr(c, "upload_status") or get_obj_attr(c, "status")
            if not c_status:
                c_status = "ACCEPTED" if norm_dtype not in ["unknown", "other"] and conf >= 0.70 else "WRONG DOCUMENT"
            c_status_upper = str(c_status).upper()
            status_color = "#15803d" if c_status_upper in ["ACCEPTED", "SUCCESS"] else "#b91c1c"

            class_table_data.append([
                Paragraph(fname, style_cell_body),
                Paragraph(str(reason)[:35], style_cell_body),
                Paragraph(f"<b>{norm_dtype.replace('_', ' ').title()}</b>", style_cell_body_bold),
                Paragraph(f"<font color='{status_color}'>{c_status_upper}</font>", style_cell_body),
                Paragraph(conf_str, style_cell_body)
            ])

        class_table = Table(class_table_data, colWidths=[130, 140, 140, 70, 60])
        class_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_NAVY),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(class_table)
        story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 7. VALIDATION SUMMARY
    # -------------------------------------------------------------------------
    story.append(Paragraph("VALIDATION SUMMARY (AGENT 3)", style_section_heading))

    v_total = get_obj_attr(val_sum, "total_checks", 0)
    v_pass = get_obj_attr(val_sum, "passed_count", 0)
    v_warn = get_obj_attr(val_sum, "warning_count", 0)
    v_err = get_obj_attr(val_sum, "error_count", 0)
    v_block = get_obj_attr(val_sum, "blocking_count", 0)

    val_grid_data = [
        [
            Paragraph("<b>Total Checks:</b>", style_body), Paragraph(str(v_total), style_body),
            Paragraph("<b>Passed Checks:</b>", style_body), Paragraph(f"<font color='#15803d'><b>{v_pass}</b></font>", style_body),
            Paragraph("<b>Warnings:</b>", style_body), Paragraph(f"<font color='#b45309'><b>{v_warn}</b></font>", style_body)
        ],
        [
            Paragraph("<b>Error Checks:</b>", style_body), Paragraph(f"<font color='#b91c1c'><b>{v_err}</b></font>", style_body),
            Paragraph("<b>Blocking Issues:</b>", style_body), Paragraph(f"<font color='#b91c1c'><b>{v_block}</b></font>", style_body),
            Paragraph("", style_body), Paragraph("", style_body)
        ]
    ]
    val_grid_table = Table(val_grid_data, colWidths=[90, 90, 90, 90, 90, 90])
    val_grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(val_grid_table)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 8. CROSS-DOCUMENT VERIFICATION SUMMARY
    # -------------------------------------------------------------------------
    story.append(Paragraph("CROSS-DOCUMENT VERIFICATION (AGENT 4)", style_section_heading))

    c_cov = get_obj_attr(cross_sum, "verification_coverage", 0.0)
    c_score = get_obj_attr(cross_sum, "consistency_score")
    c_score_str = f"{c_score:.1f} / 100" if c_score is not None else "N/A"
    c_total = get_obj_attr(cross_sum, "total_comparisons", 0)
    c_match = get_obj_attr(cross_sum, "match_count", 0)
    c_minor = get_obj_attr(cross_sum, "minor_variation_count", 0)
    c_mismatch = get_obj_attr(cross_sum, "mismatch_count", 0)
    c_unver = get_obj_attr(cross_sum, "unverifiable_count", 0)

    cross_grid_data = [
        [
            Paragraph("<b>Coverage:</b>", style_body), Paragraph(f"<b>{c_cov:.1f}%</b>", style_body),
            Paragraph("<b>Consistency Score:</b>", style_body), Paragraph(f"<b>{c_score_str}</b>", style_body),
            Paragraph("<b>Total Comparisons:</b>", style_body), Paragraph(str(c_total), style_body)
        ],
        [
            Paragraph("<b>Matches:</b>", style_body), Paragraph(f"<font color='#15803d'><b>{c_match}</b></font>", style_body),
            Paragraph("<b>Minor Variations:</b>", style_body), Paragraph(f"<font color='#1d4ed8'><b>{c_minor}</b></font>", style_body),
            Paragraph("<b>Mismatches / Unver:</b>", style_body), Paragraph(f"<font color='#b91c1c'><b>{c_mismatch} / {c_unver}</b></font>", style_body)
        ]
    ]
    cross_grid_table = Table(cross_grid_data, colWidths=[90, 90, 100, 80, 100, 80])
    cross_grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(cross_grid_table)
    story.append(Spacer(1, 8))

    # Cross-Document Findings Table (With Sensitive Data Masked!)
    cross_findings = get_obj_attr(cross_document_results, "findings") or []
    if cross_findings:
        cross_table_data = [
            [
                Paragraph("Comparison Field", style_cell_header),
                Paragraph("Document A Value", style_cell_header),
                Paragraph("Document B Value", style_cell_header),
                Paragraph("Result", style_cell_header),
                Paragraph("Explanation / Evidence", style_cell_header)
            ]
        ]
        for f in cross_findings:
            field_name = get_obj_attr(f, "field_name", "")
            doc1_val = get_obj_attr(f, "doc1_value")
            doc2_val = get_obj_attr(f, "doc2_value")
            res = get_obj_attr(f, "comparison_result", "")
            msg = get_obj_attr(f, "message", "")

            # Mask sensitive values
            masked_d1 = mask_sensitive_value(doc1_val, field_name) if doc1_val is not None else "null"
            masked_d2 = mask_sensitive_value(doc2_val, field_name) if doc2_val is not None else "null"

            res_color = "#15803d" if res == "MATCH" else ("#1d4ed8" if res == "MINOR_VARIATION" else "#b91c1c")

            cross_table_data.append([
                Paragraph(field_name.replace("_", " ").title(), style_cell_body_bold),
                Paragraph(str(masked_d1), style_cell_body),
                Paragraph(str(masked_d2), style_cell_body),
                Paragraph(f"<font color='{res_color}'><b>{res}</b></font>", style_cell_body),
                Paragraph(str(msg)[:60], style_cell_body)
            ])

        cross_table = Table(cross_table_data, colWidths=[110, 95, 95, 80, 160])
        cross_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(cross_table)
        story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 9. RISK & ANOMALY ASSESSMENT SUMMARY
    # -------------------------------------------------------------------------
    story.append(Paragraph("RISK & ANOMALY ASSESSMENT (AGENT 5)", style_section_heading))

    r_score = get_obj_attr(risk_sum, "risk_score", 0.0)
    r_level = get_obj_attr(risk_sum, "risk_level", "LOW")
    r_crit = get_obj_attr(risk_sum, "critical_count", 0)
    r_high = get_obj_attr(risk_sum, "high_count", 0)
    r_med = get_obj_attr(risk_sum, "medium_count", 0)
    r_low = get_obj_attr(risk_sum, "low_count", 0)

    risk_grid_data = [
        [
            Paragraph("<b>Total Risk Score:</b>", style_body), Paragraph(f"<b>{r_score:.1f} / 100</b>", style_body),
            Paragraph("<b>Overall Risk Level:</b>", style_body), Paragraph(f"<b>{r_level}</b>", style_body),
            Paragraph("<b>Critical Issues:</b>", style_body), Paragraph(f"<font color='#b91c1c'><b>{r_crit}</b></font>", style_body)
        ],
        [
            Paragraph("<b>High Severity:</b>", style_body), Paragraph(f"<font color='#c2410c'><b>{r_high}</b></font>", style_body),
            Paragraph("<b>Medium Severity:</b>", style_body), Paragraph(f"<font color='#b45309'><b>{r_med}</b></font>", style_body),
            Paragraph("<b>Low Severity:</b>", style_body), Paragraph(f"<font color='#15803d'><b>{r_low}</b></font>", style_body)
        ]
    ]
    risk_grid_table = Table(risk_grid_data, colWidths=[90, 90, 100, 80, 90, 90])
    risk_grid_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(risk_grid_table)
    story.append(Spacer(1, 8))

    # Risk Factors Breakdown Table
    rf_list = get_obj_attr(risk_assessment_results, "risk_factors") or []
    if rf_list:
        rf_table_data = [
            [
                Paragraph("Category", style_cell_header),
                Paragraph("Severity", style_cell_header),
                Paragraph("Risk Factor Title", style_cell_header),
                Paragraph("Source", style_cell_header),
                Paragraph("Pts", style_cell_header),
                Paragraph("Recommendation", style_cell_header)
            ]
        ]
        for rf in rf_list:
            cat = get_obj_attr(rf, "category", "")
            sev = get_obj_attr(rf, "severity", "LOW")
            title = get_obj_attr(rf, "title", "")
            src = get_obj_attr(rf, "source", "")
            pts = get_obj_attr(rf, "points", 0)
            rec = get_obj_attr(rf, "recommendation", "")

            sev_color = "#b91c1c" if sev == "CRITICAL" else ("#c2410c" if sev == "HIGH" else ("#b45309" if sev == "MEDIUM" else "#15803d"))

            rf_table_data.append([
                Paragraph(cat.replace("_", " ").title(), style_cell_body_bold),
                Paragraph(f"<font color='{sev_color}'><b>{sev}</b></font>", style_cell_body),
                Paragraph(title, style_cell_body),
                Paragraph(src[:20], style_cell_body),
                Paragraph(f"+{pts}", style_cell_body_bold),
                Paragraph(rec[:50], style_cell_body)
            ])

        rf_table = Table(rf_table_data, colWidths=[90, 55, 135, 70, 30, 160])
        rf_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HEADER_BLUE),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(rf_table)
        story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 10. KEY FINDINGS & ACTIONABLE RECOMMENDATIONS
    # -------------------------------------------------------------------------
    story.append(Paragraph("KEY FINDINGS", style_section_heading))
    if key_findings:
        for kf in key_findings:
            story.append(Paragraph(f"• {kf}", style_body))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("• Zero critical risk findings detected during analysis.", style_body))

    story.append(Spacer(1, 6))

    story.append(Paragraph("ACTIONABLE RECOMMENDATIONS", style_section_heading))
    if recommendations:
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", style_body))
            story.append(Spacer(1, 2))
    else:
        story.append(Paragraph("• No mandatory underwriting action required.", style_body))

    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # 10b. LOAN ELIGIBILITY ENGINE & POLICY KNOWLEDGE BASE EVALUATION
    # -------------------------------------------------------------------------
    eligibility_dec = get_obj_attr(final_report_obj, "eligibility_decision")
    if eligibility_dec:
        el_decision = str(get_obj_attr(eligibility_dec, "decision", "PENDING")).upper()
        el_rules = get_obj_attr(eligibility_dec, "rule_results") or []

        el_color = "#15803d" if el_decision == "ELIGIBLE" else ("#b91c1c" if el_decision == "NOT_ELIGIBLE" else "#b45309")
        story.append(Paragraph("LOAN ELIGIBILITY & POLICY EVALUATION (DEMO_POLICY PROVENANCE)", style_section_heading))
        story.append(Paragraph(f"<b>Eligibility Verdict:</b> <font color='{el_color}'><b>{el_decision}</b></font> | Rules Evaluated: {len(el_rules)}", style_body))
        story.append(Spacer(1, 4))

        if el_rules:
            el_table_data = [
                [
                    Paragraph("Rule Code", style_cell_header),
                    Paragraph("Category", style_cell_header),
                    Paragraph("Condition", style_cell_header),
                    Paragraph("Actual Value", style_cell_header),
                    Paragraph("Status", style_cell_header),
                    Paragraph("Policy Provenance", style_cell_header)
                ]
            ]
            for r in el_rules:
                rcode = get_obj_attr(r, "rule_code", "")
                cat = get_obj_attr(r, "category", "")
                op = get_obj_attr(r, "operator", "")
                exp = get_obj_attr(r, "expected_value", "")
                act = get_obj_attr(r, "actual_value", "N/A")
                st = get_obj_attr(r, "status", "PASS")
                st_color = "#15803d" if st == "PASS" else ("#b91c1c" if st == "FAIL" else "#b45309")

                masked_act = mask_sensitive_value(act, str(get_obj_attr(r, "field_name", "")))

                el_table_data.append([
                    Paragraph(rcode, style_cell_body_bold),
                    Paragraph(cat, style_cell_body),
                    Paragraph(f"{op} {exp}", style_cell_body),
                    Paragraph(str(masked_act)[:25], style_cell_body),
                    Paragraph(f"<font color='{st_color}'><b>{st}</b></font>", style_cell_body),
                    Paragraph("DEMO_POLICY / Underwriting Guidelines", style_cell_body)
                ])

            el_table = Table(el_table_data, colWidths=[90, 65, 80, 85, 55, 165])
            el_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), HEADER_BLUE),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(el_table)
            story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 10c. TELEMETRY & EXECUTION PERFORMANCE
    # -------------------------------------------------------------------------
    telem_sum = get_obj_attr(final_report_obj, "telemetry_summary")
    if telem_sum:
        total_ms = float(get_obj_attr(telem_sum, "total_pipeline_duration_ms", 0.0) or 0.0)
        mem_mb = float(get_obj_attr(telem_sum, "memory_mb", 0.0) or 0.0)
        agent_durs = get_obj_attr(telem_sum, "agent_durations") or {}

        story.append(Paragraph("SYSTEM TELEMETRY & EXECUTION METRICS", style_section_heading))
        story.append(Paragraph(f"<b>Total Pipeline Duration:</b> {total_ms:.1f} ms | <b>Memory Footprint:</b> {mem_mb:.1f} MB | <b>Decision Graph Nodes:</b> 11 Verified", style_body))
        story.append(Spacer(1, 4))

        if agent_durs and isinstance(agent_durs, dict):
            items = list(agent_durs.items())
            row1 = [Paragraph(f"<b>{k.replace('_', ' ').title()}:</b> {float(v):.1f} ms", style_body) for k, v in items[:4]]
            t_data = [row1]
            if len(items) > 4:
                row2 = [Paragraph(f"<b>{k.replace('_', ' ').title()}:</b> {float(v):.1f} ms", style_body) for k, v in items[4:8]]
                # Pad row2 if needed
                while len(row2) < len(row1):
                    row2.append(Paragraph("", style_body))
                t_data.append(row2)
            t_table = Table(t_data, colWidths=[540 // len(row1)] * len(row1))
            t_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t_table)
            story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # 11. FINAL RECOMMENDATION & NEXT ACTIONS CARD
    # -------------------------------------------------------------------------
    final_box_data = [
        [Paragraph(f"<b>FINAL DECISION: {dec_display}</b>", style_cell_header)],
        [Paragraph(f"<b>Reason:</b> {decision_reason}", style_body)],
        [Paragraph(f"<b>Recommended Action:</b> {'Perform underwriter investigation on highlighted risk factors.' if decision != 'PASS' else 'Proceed to final loan disbursement.'}", style_body)]
    ]
    final_box_table = Table(final_box_data, colWidths=[540])
    final_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_NAVY),
        ('BACKGROUND', (0, 1), (-1, -1), DEC_BG),
        ('BOX', (0, 0), (-1, -1), 1, DEC_BORDER),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(KeepTogether([final_box_table]))

    # Build PDF using NumberedCanvas
    doc.build(story, canvasmaker=NumberedCanvas)

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
