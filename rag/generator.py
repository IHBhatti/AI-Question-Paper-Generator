"""
rag/generator.py
-----------------
Orchestrates the actual RAG generation step: retrieve relevant chunks,
build the correct prompt, call the free Groq API, and parse the
JSON response into clean Python structures the UI/PDF export can use.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, GROQ_MAX_TOKENS, GROQ_MODEL, GROQ_TEMPERATURE
from rag.prompts import (
    ANSWER_KEY_PROMPT_TEMPLATE,
    LONG_QUESTION_PROMPT_TEMPLATE,
    MCQ_PROMPT_TEMPLATE,
    MIXED_PAPER_PROMPT_TEMPLATE,
    SHORT_QUESTION_PROMPT_TEMPLATE,
    build_topic_queries,
)
from rag.retriever import format_context, retrieve_diverse_sample

logger = logging.getLogger(__name__)

_llm: Optional[ChatGroq] = None


class InsufficientContextError(Exception):
    """Raised when the LLM reports the retrieved context can't support the
    requested question generation, so the caller can surface a clear
    message instead of a fabricated paper."""


def get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and set it as an environment "
                "variable or in a .env file."
            )
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )
    return _llm


def _extract_json(raw_text: str) -> dict:
    """Groq/Llama models occasionally wrap JSON in code fences or add
    stray text despite instructions. This strips that defensively."""
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # If there's leading/trailing prose, grab the outermost {...} block.
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM JSON output: %s\nRaw: %s", e, raw_text[:500])
        raise


def _call_llm(prompt: str) -> dict:
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _extract_json(content)


def _normalize_mcq(q: dict) -> dict:
    q.setdefault("options", {})
    q.setdefault("correct_option", "")
    q.setdefault("explanation", "")
    return q


def _normalize_short(q: dict) -> dict:
    q.setdefault("model_answer", "")
    return q


def _normalize_long(q: dict) -> dict:
    # model_answer is the actual full written answer a student would
    # study from; marking_points is separate, shorter grading guidance.
    # Neither should be silently missing — default to empty rather than
    # leaving the key absent, so downstream code can reliably check them.
    q.setdefault("model_answer", "")
    if not q.get("marking_points"):
        q["marking_points"] = []
    return q


def _normalize_result(result: dict, question_type: str) -> dict:
    """Guarantee every question dict has the fields the rest of the app
    expects, regardless of what the LLM actually returned. Also logs a
    warning (does not raise) if the model returned fewer questions than
    requested, since Groq/Llama models don't always follow exact counts."""
    if question_type == "Mixed Paper":
        result["mcqs"] = [_normalize_mcq(q) for q in result.get("mcqs", [])]
        result["short_questions"] = [_normalize_short(q) for q in result.get("short_questions", [])]
        result["long_questions"] = [_normalize_long(q) for q in result.get("long_questions", [])]
    elif question_type == "MCQs":
        result["questions"] = [_normalize_mcq(q) for q in result.get("questions", [])]
    elif question_type == "Short Questions":
        result["questions"] = [_normalize_short(q) for q in result.get("questions", [])]
    elif question_type == "Long Questions":
        result["questions"] = [_normalize_long(q) for q in result.get("questions", [])]
    return result


def generate_questions(
    vectorstore,
    subject: str,
    chapters: List[str],
    question_type: str,
    difficulty: str,
    bloom_levels: List[str],
    num_questions: int,
    mixed_split: Optional[Dict[str, int]] = None,
) -> dict:
    """Main entry point used by app.py. Retrieves context and generates
    questions of the requested type. Raises InsufficientContextError if
    the model reports the retrieved material can't support the request."""

    bloom_str = ", ".join(bloom_levels) if bloom_levels else "any"

    if question_type == "Mixed Paper":
        split = mixed_split or {}
        num_mcq = split.get("mcq", max(1, int(num_questions * 0.6)))
        num_short = split.get("short", max(1, int(num_questions * 0.25)))
        num_long = split.get("long", max(1, num_questions - num_mcq - num_short))

        topics = build_topic_queries(subject, chapters, bloom_levels, num_mcq + num_short + num_long)
        chunks = retrieve_diverse_sample(vectorstore, topics, chapters=chapters)
        context = format_context(chunks)

        prompt = MIXED_PAPER_PROMPT_TEMPLATE.format(
            num_mcq=num_mcq,
            num_short=num_short,
            num_long=num_long,
            difficulty=difficulty,
            bloom_levels=bloom_str,
            context=context,
        )
        result = _call_llm(prompt)

    else:
        topics = build_topic_queries(subject, chapters, bloom_levels, num_questions)
        chunks = retrieve_diverse_sample(vectorstore, topics, chapters=chapters)
        context = format_context(chunks)

        template = {
            "MCQs": MCQ_PROMPT_TEMPLATE,
            "Short Questions": SHORT_QUESTION_PROMPT_TEMPLATE,
            "Long Questions": LONG_QUESTION_PROMPT_TEMPLATE,
        }.get(question_type)

        if template is None:
            raise ValueError(f"Unknown question_type: {question_type}")

        prompt = template.format(
            num_questions=num_questions,
            difficulty=difficulty,
            bloom_levels=bloom_str,
            context=context,
        )
        result = _call_llm(prompt)

    if result.get("insufficient_context"):
        raise InsufficientContextError(
            "The uploaded document does not contain enough information to "
            "generate the requested questions for the selected chapters. "
            "Try selecting more chapters, the entire document, or reducing "
            "the number of questions requested."
        )

    result = _normalize_result(result, question_type)

    # Groq/Llama models don't always return exactly the count requested —
    # this is a known LLM compliance gap, most often seen with long
    # questions. Log it clearly so it's easy to spot in troubleshooting,
    # rather than the shortfall silently showing up only much later as a
    # thin-looking answer key.
    if question_type == "Mixed Paper":
        got_mcq, got_short, got_long = (
            len(result.get("mcqs", [])),
            len(result.get("short_questions", [])),
            len(result.get("long_questions", [])),
        )
        if (num_mcq, num_short, num_long) != (got_mcq, got_short, got_long):
            logger.warning(
                "Mixed paper count mismatch — requested MCQ=%d/Short=%d/Long=%d, "
                "got MCQ=%d/Short=%d/Long=%d",
                num_mcq, num_short, num_long, got_mcq, got_short, got_long,
            )
    else:
        got = len(result.get("questions", []))
        if got != num_questions:
            logger.warning(
                "%s count mismatch — requested %d, got %d", question_type, num_questions, got
            )

    return result


def generate_answer_key(vectorstore, questions: dict, chapters: List[str]) -> dict:
    """Generate a verified answer key for an already-generated question set."""
    # Build a broad context spanning the same chapters used for generation.
    topics = build_topic_queries("answer key verification", chapters, [], 6)
    chunks = retrieve_diverse_sample(vectorstore, topics, chapters=chapters)
    context = format_context(chunks)

    prompt = ANSWER_KEY_PROMPT_TEMPLATE.format(
        questions_json=json.dumps(questions, ensure_ascii=False),
        context=context,
    )
    result = _call_llm(prompt)

    if result.get("insufficient_context"):
        raise InsufficientContextError(
            "Could not verify answers against the document context."
        )

    return result


def merge_answer_key(
    mcqs: List[dict],
    short_qs: List[dict],
    long_qs: List[dict],
    key_result: Optional[dict],
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Fill in any missing/empty answer fields (correct_option/explanation,
    model_answer, marking_points) using the verified answer key from
    `generate_answer_key`, without ever discarding or overwriting content
    that was already present from the original question generation.

    This is what actually makes the "verify answers against source
    material" step meaningful — previously its result was computed and
    then thrown away, so any gaps left by the initial generation (e.g. a
    long question with no marking_points) never got backfilled.
    """
    mcqs = copy.deepcopy(mcqs)
    short_qs = copy.deepcopy(short_qs)
    long_qs = copy.deepcopy(long_qs)

    if not key_result or not key_result.get("answer_key"):
        return mcqs, short_qs, long_qs

    # The verification prompt returns one flat list ordered the same way
    # the questions were presented to it (MCQs, then short, then long) —
    # track a running index per type to map entries back correctly.
    counters = {"MCQ": 0, "Short": 0, "Long": 0}

    for entry in key_result["answer_key"]:
        q_type = entry.get("type")
        answer_text = (entry.get("answer") or "").strip()
        if not answer_text or q_type not in counters:
            continue
        idx = counters[q_type]

        if q_type == "MCQ" and idx < len(mcqs):
            if not mcqs[idx].get("explanation"):
                mcqs[idx]["explanation"] = answer_text
            if not mcqs[idx].get("correct_option") and len(answer_text) == 1:
                mcqs[idx]["correct_option"] = answer_text.upper()
        elif q_type == "Short" and idx < len(short_qs):
            if not short_qs[idx].get("model_answer"):
                short_qs[idx]["model_answer"] = answer_text
        elif q_type == "Long" and idx < len(long_qs):
            if not long_qs[idx].get("model_answer"):
                long_qs[idx]["model_answer"] = answer_text
            if not long_qs[idx].get("marking_points"):
                long_qs[idx]["marking_points"] = [answer_text]

        counters[q_type] = idx + 1

    return mcqs, short_qs, long_qs
