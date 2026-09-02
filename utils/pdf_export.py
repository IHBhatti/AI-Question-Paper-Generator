"""
utils/pdf_export.py
--------------------
Renders a generated test paper (and, separately, its answer key) into a
clean, professional PDF using ReportLab — a free, open-source library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
from xml.sax.saxutils import escape as _xml_escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import OUTPUTS_DIR

styles = getSampleStyleSheet()

TITLE_STYLE = ParagraphStyle(
    "PaperTitle", parent=styles["Title"], fontSize=18, alignment=TA_CENTER, spaceAfter=4
)
SUBTITLE_STYLE = ParagraphStyle(
    "PaperSubtitle", parent=styles["Normal"], fontSize=11, alignment=TA_CENTER, textColor=colors.grey
)
META_STYLE = ParagraphStyle(
    "Meta", parent=styles["Normal"], fontSize=10, alignment=TA_LEFT, spaceAfter=2
)
SECTION_HEADER_STYLE = ParagraphStyle(
    "SectionHeader",
    parent=styles["Heading2"],
    fontSize=13,
    spaceBefore=14,
    spaceAfter=8,
    textColor=colors.HexColor("#1a1a2e"),
)
QUESTION_STYLE = ParagraphStyle(
    "Question", parent=styles["Normal"], fontSize=10.5, spaceAfter=8, leading=15
)
OPTION_STYLE = ParagraphStyle(
    "Option", parent=styles["Normal"], fontSize=10, leftIndent=18, spaceAfter=2, leading=13
)
INSTRUCTIONS_STYLE = ParagraphStyle(
    "Instructions", parent=styles["Normal"], fontSize=9.5, leftIndent=10, spaceAfter=2
)
FOOTER_STYLE = ParagraphStyle(
    "Footer", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER, textColor=colors.grey
)


def _esc(text) -> str:
    """Escape text before inserting it into a ReportLab Paragraph.

    Paragraph interprets its input as a small HTML-like markup language
    (it supports tags like <b>, <i>). Without escaping, LLM- or
    OCR-generated content containing '&', '<', or '>' — extremely common
    in STEM subjects (chemical equations, inequality symbols, "R&D",
    set notation, etc.) — gets silently misinterpreted as malformed
    markup. Confirmed directly: an unescaped "R&D and A&B" rendered in
    the actual PDF as "R&D; and A&B;" with spurious semicolons inserted,
    not a crash but silently wrong content. Every piece of dynamic text
    must be escaped before being embedded in an f-string passed to
    Paragraph — literal tags we add ourselves (e.g. "<b>") are written
    outside of this and stay as real markup.
    """
    if text is None:
        return ""
    return _xml_escape(str(text))


def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(A4[0] / 2, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def _header_block(meta: Dict) -> list:
    elements = []
    if meta.get("institution_name"):
        elements.append(Paragraph(_esc(meta["institution_name"]), TITLE_STYLE))
    elements.append(Paragraph(_esc(meta.get("subject", "Examination Paper")), SUBTITLE_STYLE))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    elements.append(Spacer(1, 8))

    info_rows = [
        ["Subject:", _esc(meta.get("subject", "-")), "Total Marks:", _esc(meta.get("total_marks", "-"))],
        ["Exam Date:", _esc(meta.get("exam_date", "-")), "Time Duration:", _esc(meta.get("time_duration", "-"))],
        ["Chapters Covered:", _esc(", ".join(meta.get("chapters", [])) or "Entire Document"), "", ""],
    ]
    table = Table(info_rows, colWidths=[3.2 * cm, 6 * cm, 3.2 * cm, 4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 8))

    if meta.get("instructions"):
        elements.append(Paragraph("<b>Instructions</b>", META_STYLE))
        for line in meta["instructions"].split("\n"):
            if line.strip():
                elements.append(Paragraph(f"• {_esc(line.strip())}", INSTRUCTIONS_STYLE))
        elements.append(Spacer(1, 8))

    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    elements.append(Spacer(1, 10))
    return elements


def _mcq_section(mcqs: List[dict], marks_per_question: int) -> list:
    elements = [Paragraph(f"Section A — Multiple Choice Questions ({marks_per_question} mark each)", SECTION_HEADER_STYLE)]
    for i, q in enumerate(mcqs, start=1):
        elements.append(Paragraph(f"{i}. {_esc(q['question'])}", QUESTION_STYLE))
        for letter, option_text in q.get("options", {}).items():
            elements.append(Paragraph(f"({_esc(letter)}) {_esc(option_text)}", OPTION_STYLE))
        elements.append(Spacer(1, 4))
    return elements


def _short_section(questions: List[dict], marks_per_question: int) -> list:
    elements = [Paragraph(f"Section B — Short Questions ({marks_per_question} marks each)", SECTION_HEADER_STYLE)]
    for i, q in enumerate(questions, start=1):
        elements.append(Paragraph(f"{i}. {_esc(q['question'])}", QUESTION_STYLE))
    return elements


def _long_section(questions: List[dict], marks_per_question: int) -> list:
    elements = [Paragraph(f"Section C — Long Questions ({marks_per_question} marks each)", SECTION_HEADER_STYLE)]
    for i, q in enumerate(questions, start=1):
        elements.append(Paragraph(f"{i}. {_esc(q['question'])}", QUESTION_STYLE))
    return elements


def export_test_paper_pdf(
    meta: Dict,
    mcqs: Optional[List[dict]] = None,
    short_questions: Optional[List[dict]] = None,
    long_questions: Optional[List[dict]] = None,
    marks: Optional[Dict[str, int]] = None,
    output_filename: str = "test_paper.pdf",
) -> Path:
    """Build the full exam-paper PDF and return the output file path."""
    marks = marks or {"MCQ": 1, "Short": 5, "Long": 10}
    output_path = OUTPUTS_DIR / output_filename

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title=meta.get("subject", "Test Paper"),
    )

    story = []
    story.extend(_header_block(meta))

    if mcqs:
        story.extend(_mcq_section(mcqs, marks.get("MCQ", 1)))
        story.append(Spacer(1, 6))
    if short_questions:
        story.extend(_short_section(short_questions, marks.get("Short", 5)))
        story.append(Spacer(1, 6))
    if long_questions:
        story.extend(_long_section(long_questions, marks.get("Long", 10)))

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")))
    story.append(Spacer(1, 6))
    story.append(Paragraph("— End of Paper —", FOOTER_STYLE))

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return output_path


def export_answer_key_pdf(
    meta: Dict,
    mcqs: Optional[List[dict]] = None,
    short_questions: Optional[List[dict]] = None,
    long_questions: Optional[List[dict]] = None,
    output_filename: str = "answer_key.pdf",
) -> Path:
    """Build a separate answer-key PDF."""
    output_path = OUTPUTS_DIR / output_filename

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.8 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        title=f"Answer Key - {meta.get('subject', '')}",
    )

    story = [
        Paragraph(f"Answer Key — {_esc(meta.get('subject', ''))}", TITLE_STYLE),
        Spacer(1, 10),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a1a2e")),
        Spacer(1, 10),
    ]

    counter = 1
    if mcqs:
        story.append(Paragraph("Section A — MCQ Answers", SECTION_HEADER_STYLE))
        for q in mcqs:
            story.append(Paragraph(f"<b>{counter}. {_esc(q.get('question', ''))}</b>", QUESTION_STYLE))
            story.append(
                Paragraph(
                    f"Correct Option: <b>{_esc(q.get('correct_option', '-'))}</b> — {_esc(q.get('explanation', '-'))}",
                    OPTION_STYLE,
                )
            )
            story.append(Spacer(1, 6))
            counter += 1
        counter = 1

    if short_questions:
        story.append(Paragraph("Section B — Model Answers", SECTION_HEADER_STYLE))
        for q in short_questions:
            story.append(Paragraph(f"<b>{counter}. {_esc(q.get('question', ''))}</b>", QUESTION_STYLE))
            story.append(Paragraph(_esc(q.get("model_answer", "-") or "-"), OPTION_STYLE))
            story.append(Spacer(1, 6))
            counter += 1
        counter = 1

    if long_questions:
        story.append(Paragraph("Section C — Model Answers", SECTION_HEADER_STYLE))
        for q in long_questions:
            story.append(Paragraph(f"<b>{counter}. {_esc(q.get('question', ''))}</b>", QUESTION_STYLE))
            story.append(Paragraph(_esc(q.get("model_answer", "-") or "-"), OPTION_STYLE))
            points = q.get("marking_points", [])
            if points:
                story.append(Spacer(1, 3))
                escaped_points = "; ".join(_esc(p) for p in points)
                story.append(Paragraph(f"<i>Key marking points: {escaped_points}</i>", OPTION_STYLE))
            story.append(Spacer(1, 8))
            counter += 1

    doc.build(story, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return output_path
