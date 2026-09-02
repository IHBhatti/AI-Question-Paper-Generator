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
import time
from typing import Dict, List, Optional, Tuple

from langchain_groq import ChatGroq

from config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    MAX_CONTEXT_CHUNKS_PER_CALL,
    MAX_LONG_PER_CALL,
    MAX_MCQ_PER_CALL,
    MAX_SHORT_PER_CALL,
)
from rag.prompts import (
    ANSWER_KEY_PROMPT_TEMPLATE,
    LONG_QUESTION_PROMPT_TEMPLATE,
    MCQ_PROMPT_TEMPLATE,
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


class TruncatedResponseError(Exception):
    """Raised when the LLM's JSON response was cut off mid-output —
    almost always because it hit the token limit before finishing,
    typically from requesting too many questions (especially long
    questions, which require full written answers) in one call."""


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
        # An "Unterminated string" or "Expecting ',' delimiter" error at
        # the very end of the text is the signature of a response that
        # got cut off mid-generation (hit max_tokens) rather than a
        # genuinely malformed response — worth telling the person that
        # directly, since the fix (fewer questions, or raise
        # GROQ_MAX_TOKENS in config.py) is different from a generic bug.
        near_the_end = e.pos >= len(text) - 20
        if "Unterminated string" in str(e) or near_the_end:
            raise TruncatedResponseError(
                "The AI's response was cut off before it finished (it ran out of "
                "output space). Try requesting fewer questions in one go, or — if "
                "you're editing config.py — increase GROQ_MAX_TOKENS."
            ) from e
        raise


class RateLimitedError(Exception):
    """Raised when Groq's rate limit (requests-per-minute or
    tokens-per-minute) is hit repeatedly even after retrying — surfaced
    with a clear, actionable message instead of a raw HTTP error."""


_RATE_LIMIT_MARKERS = ("429", "rate_limit", "rate limit", "too many requests")
_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 12  # Groq's per-minute limits reset on a rolling window


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS)


def _call_llm(prompt: str) -> dict:
    llm = get_llm()
    last_error: Optional[Exception] = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            return _extract_json(content)
        except (TruncatedResponseError, json.JSONDecodeError):
            raise
        except Exception as e:
            if not _is_rate_limit_error(e):
                raise
            last_error = e
            if attempt < _MAX_RETRIES:
                logger.warning(
                    "Rate limited by Groq (attempt %d/%d) — waiting %ds before retrying",
                    attempt, _MAX_RETRIES, _RETRY_DELAY_SECONDS,
                )
                time.sleep(_RETRY_DELAY_SECONDS)

    raise RateLimitedError(
        "Groq's rate limit was hit repeatedly and retrying didn't help. This "
        "usually means too many requests or tokens were used in a short "
        "window (common on the free tier with large papers). Try again in "
        "a minute, request fewer questions, or check "
        "https://console.groq.com/settings/billing for your current limits."
    ) from last_error


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


_BATCH_CAP_BY_TYPE = {
    "MCQs": MAX_MCQ_PER_CALL,
    "Short Questions": MAX_SHORT_PER_CALL,
    "Long Questions": MAX_LONG_PER_CALL,
}
_TEMPLATE_BY_TYPE = {
    "MCQs": MCQ_PROMPT_TEMPLATE,
    "Short Questions": SHORT_QUESTION_PROMPT_TEMPLATE,
    "Long Questions": LONG_QUESTION_PROMPT_TEMPLATE,
}


def _generate_one_batch(
    vectorstore,
    subject: str,
    effective_chapters: List[str],
    question_type: str,
    difficulty: str,
    bloom_levels: List[str],
    count: int,
) -> List[dict]:
    """Run a single LLM call for up to `count` questions of one type,
    with context capped at MAX_CONTEXT_CHUNKS_PER_CALL so the prompt side
    of the request stays bounded too. Returns the normalized question
    list (never raises InsufficientContextError for an empty batch —
    that's the caller's call to make once all batches are combined)."""
    bloom_str = ", ".join(bloom_levels) if bloom_levels else "any"
    topic_pairs = build_topic_queries(subject, effective_chapters, bloom_levels, count)
    chunks = retrieve_diverse_sample(vectorstore, topic_pairs)[:MAX_CONTEXT_CHUNKS_PER_CALL]
    context = format_context(chunks)

    template = _TEMPLATE_BY_TYPE[question_type]
    prompt = template.format(
        num_questions=count, difficulty=difficulty, bloom_levels=bloom_str, context=context
    )
    result = _call_llm(prompt)

    if result.get("insufficient_context"):
        return []

    result = _normalize_result({"questions": result.get("questions", [])}, question_type)
    return result["questions"][:count]


def _generate_batched_type(
    vectorstore,
    subject: str,
    effective_chapters: List[str],
    question_type: str,
    difficulty: str,
    bloom_levels: List[str],
    total_count: int,
) -> List[dict]:
    """Generate `total_count` questions of one type, transparently split
    across as many smaller LLM calls as needed to keep every individual
    request safely within Groq's per-minute token budget — rather than
    one large call whose max_tokens requirement alone can exceed the
    free-tier limit (this is what caused a 413 "Request too large"
    error previously, since a single call for a full mixed paper with
    several full-prose long-question answers could need more tokens per
    minute than the account's tier allows).
    """
    if total_count <= 0:
        return []

    batch_cap = _BATCH_CAP_BY_TYPE[question_type]
    collected: List[dict] = []
    batch_num = 0

    while len(collected) < total_count:
        if batch_num > 0:
            # A brief pause between successive calls for the same paper
            # reduces the chance of tripping Groq's requests-per-minute
            # limit in the first place, on top of the retry-with-backoff
            # already handled inside _call_llm if it happens anyway.
            time.sleep(1.5)
        batch_num += 1

        remaining = total_count - len(collected)
        batch_size = min(batch_cap, remaining)
        batch = _generate_one_batch(
            vectorstore, subject, effective_chapters, question_type, difficulty, bloom_levels, batch_size
        )
        if not batch:
            # A batch coming back empty means the model reported
            # insufficient context for that slice — stop asking for more
            # rather than looping forever; whatever was already collected
            # (possibly nothing) is returned as-is.
            break
        collected.extend(batch)

    return collected[:total_count]


def generate_questions(
    vectorstore,
    subject: str,
    chapters: List[str],
    question_type: str,
    difficulty: str,
    bloom_levels: List[str],
    num_questions: int,
    mixed_split: Optional[Dict[str, int]] = None,
    all_available_chapters: Optional[List[str]] = None,
) -> dict:
    """Main entry point used by app.py. Retrieves context and generates
    questions of the requested type. Raises InsufficientContextError if
    the model reports the retrieved material can't support the request.

    `chapters` is the user's explicit selection. When empty (the person
    chose "Entire Document"), `all_available_chapters` -- the full list of
    chapters actually detected in the document -- is used instead, so
    retrieval still round-robins across every chapter rather than
    defaulting to an unrestricted search that tends to be dominated by
    whichever chapter scores highest on generic semantic similarity.

    Generation is internally batched per question type (see
    _generate_batched_type) rather than requested as one giant call, so
    a large paper never needs more tokens in a single request than
    Groq's free-tier per-minute budget allows.
    """
    effective_chapters = chapters if chapters else (all_available_chapters or [])

    if question_type == "Mixed Paper":
        split = mixed_split or {}
        num_mcq = split.get("mcq", max(1, int(num_questions * 0.6)))
        num_short = split.get("short", max(1, int(num_questions * 0.25)))
        num_long = split.get("long", max(1, num_questions - num_mcq - num_short))

        mcqs = _generate_batched_type(
            vectorstore, subject, effective_chapters, "MCQs", difficulty, bloom_levels, num_mcq
        )
        short_questions = _generate_batched_type(
            vectorstore, subject, effective_chapters, "Short Questions", difficulty, bloom_levels, num_short
        )
        long_questions = _generate_batched_type(
            vectorstore, subject, effective_chapters, "Long Questions", difficulty, bloom_levels, num_long
        )

        if not (mcqs or short_questions or long_questions):
            raise InsufficientContextError(
                "The uploaded document does not contain enough information to "
                "generate the requested questions for the selected chapters. "
                "Try selecting more chapters, the entire document, or reducing "
                "the number of questions requested."
            )

        for label, requested, got_list in (
            ("MCQ", num_mcq, mcqs), ("Short", num_short, short_questions), ("Long", num_long, long_questions)
        ):
            if len(got_list) != requested:
                logger.warning(
                    "Mixed paper %s count mismatch -- requested %d, got %d "
                    "(the model returned fewer than requested for this section)",
                    label, requested, len(got_list),
                )

        return {
            "insufficient_context": False,
            "mcqs": mcqs,
            "short_questions": short_questions,
            "long_questions": long_questions,
        }

    else:
        if question_type not in _TEMPLATE_BY_TYPE:
            raise ValueError(f"Unknown question_type: {question_type}")

        questions = _generate_batched_type(
            vectorstore, subject, effective_chapters, question_type, difficulty, bloom_levels, num_questions
        )

        if not questions:
            raise InsufficientContextError(
                "The uploaded document does not contain enough information to "
                "generate the requested questions for the selected chapters. "
                "Try selecting more chapters, the entire document, or reducing "
                "the number of questions requested."
            )

        if len(questions) != num_questions:
            logger.warning(
                "%s count mismatch -- requested %d, got %d (the model returned "
                "fewer than requested)",
                question_type, num_questions, len(questions),
            )

        return {"insufficient_context": False, "questions": questions}


def generate_answer_key(vectorstore, questions: dict, chapters: List[str]) -> dict:
    """Generate a verified answer key for an already-generated question set."""
    # Build a broad context spanning the same chapters used for generation.
    topic_pairs = build_topic_queries("answer key verification", chapters, [], 6)
    chunks = retrieve_diverse_sample(vectorstore, topic_pairs)
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
