"""
utils/text_export.py
---------------------
Renders the generated test paper (and answer key) as plain .txt files,
as an alternative to the PDF export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from config import OUTPUTS_DIR


def _wrap_header(meta: Dict) -> str:
    lines = []
    if meta.get("institution_name"):
        lines.append(meta["institution_name"].upper())
    lines.append(meta.get("subject", "Examination Paper"))
    lines.append("=" * 60)
    lines.append(f"Subject        : {meta.get('subject', '-')}")
    lines.append(f"Exam Date      : {meta.get('exam_date', '-')}")
    lines.append(f"Time Duration  : {meta.get('time_duration', '-')}")
    lines.append(f"Total Marks    : {meta.get('total_marks', '-')}")
    chapters = meta.get("chapters", [])
    lines.append(f"Chapters       : {', '.join(chapters) if chapters else 'Entire Document'}")
    lines.append("-" * 60)
    if meta.get("instructions"):
        lines.append("Instructions:")
        for line in meta["instructions"].split("\n"):
            if line.strip():
                lines.append(f"  • {line.strip()}")
        lines.append("-" * 60)
    return "\n".join(lines)


def export_test_paper_txt(
    meta: Dict,
    mcqs: Optional[List[dict]] = None,
    short_questions: Optional[List[dict]] = None,
    long_questions: Optional[List[dict]] = None,
    marks: Optional[Dict[str, int]] = None,
    output_filename: str = "test_paper.txt",
) -> Path:
    marks = marks or {"MCQ": 1, "Short": 5, "Long": 10}
    output_path = OUTPUTS_DIR / output_filename

    lines = [_wrap_header(meta), ""]

    if mcqs:
        lines.append(f"SECTION A — MULTIPLE CHOICE QUESTIONS ({marks.get('MCQ', 1)} mark each)")
        lines.append("-" * 60)
        for i, q in enumerate(mcqs, start=1):
            lines.append(f"{i}. {q['question']}")
            for letter, opt in q.get("options", {}).items():
                lines.append(f"   ({letter}) {opt}")
            lines.append("")

    if short_questions:
        lines.append(f"SECTION B — SHORT QUESTIONS ({marks.get('Short', 5)} marks each)")
        lines.append("-" * 60)
        for i, q in enumerate(short_questions, start=1):
            lines.append(f"{i}. {q['question']}")
            lines.append("")

    if long_questions:
        lines.append(f"SECTION C — LONG QUESTIONS ({marks.get('Long', 10)} marks each)")
        lines.append("-" * 60)
        for i, q in enumerate(long_questions, start=1):
            lines.append(f"{i}. {q['question']}")
            lines.append("")

    lines.append("=" * 60)
    lines.append("END OF PAPER".center(60))

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_answer_key_txt(
    meta: Dict,
    mcqs: Optional[List[dict]] = None,
    short_questions: Optional[List[dict]] = None,
    long_questions: Optional[List[dict]] = None,
    output_filename: str = "answer_key.txt",
) -> Path:
    output_path = OUTPUTS_DIR / output_filename
    lines = [f"ANSWER KEY — {meta.get('subject', '')}", "=" * 60, ""]

    if mcqs:
        lines.append("SECTION A — MCQ ANSWERS")
        lines.append("-" * 60)
        for i, q in enumerate(mcqs, start=1):
            lines.append(f"{i}. {q.get('question', '')}")
            lines.append(f"   Correct Option: {q.get('correct_option', '-')}")
            lines.append(f"   Explanation: {q.get('explanation', '-')}")
            lines.append("")
        lines.append("")

    if short_questions:
        lines.append("SECTION B — MODEL ANSWERS")
        lines.append("-" * 60)
        for i, q in enumerate(short_questions, start=1):
            lines.append(f"{i}. {q.get('question', '')}")
            lines.append(f"   {q.get('model_answer', '-') or '-'}")
            lines.append("")
        lines.append("")

    if long_questions:
        lines.append("SECTION C — MODEL ANSWERS")
        lines.append("-" * 60)
        for i, q in enumerate(long_questions, start=1):
            lines.append(f"{i}. {q.get('question', '')}")
            lines.append(f"   {q.get('model_answer', '-') or '-'}")
            points = q.get("marking_points", [])
            if points:
                lines.append(f"   Key marking points: {'; '.join(points)}")
            lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
