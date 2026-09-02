"""
rag/prompts.py
--------------
All prompt templates used to drive the Groq LLM. Every template
explicitly restricts the model to the supplied context and instructs it
to say so plainly if the context is insufficient, rather than inventing
content from general knowledge.
"""

GROUNDING_RULES = """\
You are an exam-question generator operating under strict retrieval-augmented
generation rules.

RULES (must follow exactly):
1. You may ONLY use facts, definitions, examples, and explanations that
   appear in the CONTEXT below. Do not use any outside/general knowledge.
2. If the CONTEXT does not contain enough information to create the
   requested number or type of questions, say so explicitly in the
   "insufficient_context" field instead of inventing content.
3. Every question must be answerable using only the CONTEXT provided.
4. Do not mention "the context" or "the document" inside the question text
   itself — write questions as a real exam paper would.
5. Return ONLY valid JSON matching the schema described. No prose,
   no markdown code fences, no commentary before or after the JSON.
6. Return EXACTLY the number of questions requested for each question
   type — not fewer, not more. If you cannot reach the exact count from
   the given context, generate as many as the context genuinely supports
   and set "insufficient_context" to true rather than silently returning
   fewer questions with no explanation.
7. Every field in the schema is required and must never be left empty
   or omitted — in particular, "model_answer" for long questions must be
   a complete, fully written answer (never blank, never just a list of
   fragments), "marking_points" for long questions must always contain
   3 to 5 concrete points as separate grading guidance, and "model_answer"
   for short questions must never be blank.
8. The CONTEXT below may span multiple chapters (each chunk is labeled
   with its chapter). Distribute questions as evenly as possible across
   every chapter actually represented in the CONTEXT — do not let
   questions cluster only in the first chapter or two just because that
   material appears first.
"""

MCQ_PROMPT_TEMPLATE = GROUNDING_RULES + """
TASK: Generate {num_questions} Multiple Choice Questions (MCQs) at
"{difficulty}" difficulty, targeting Bloom's Taxonomy level(s): {bloom_levels}.

CONTEXT:
{context}

Return JSON with this exact schema:
{{
  "insufficient_context": false,
  "questions": [
    {{
      "question": "string",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "brief explanation of why the correct option is right",
      "difficulty": "Easy" | "Medium" | "Hard",
      "bloom_level": "string",
      "source_chapter": "string",
      "source_page": "string"
    }}
  ]
}}
"""

SHORT_QUESTION_PROMPT_TEMPLATE = GROUNDING_RULES + """
TASK: Generate {num_questions} Short-Answer Questions at "{difficulty}"
difficulty, targeting Bloom's Taxonomy level(s): {bloom_levels}.
Expected answer length is 2-5 lines per question.

CONTEXT:
{context}

Return JSON with this exact schema:
{{
  "insufficient_context": false,
  "questions": [
    {{
      "question": "string",
      "model_answer": "a concise 2-5 line model answer",
      "difficulty": "Easy" | "Medium" | "Hard",
      "bloom_level": "string",
      "source_chapter": "string",
      "source_page": "string"
    }}
  ]
}}
"""

LONG_QUESTION_PROMPT_TEMPLATE = GROUNDING_RULES + """
TASK: Generate {num_questions} Long-Answer Questions at "{difficulty}"
difficulty, targeting Bloom's Taxonomy level(s): {bloom_levels}.
Questions should require explanation, analysis, and/or application, with an
expected answer length of 200-500 words.

For each question, write a COMPLETE, FULLY WRITTEN model answer in flowing
prose (not just a list of fragments) that actually addresses every part of
the question — if the question asks for a definition AND examples AND a
comparison, the model_answer must cover all of those, in that order, as
real sentences and paragraphs a student could study from directly.
"marking_points" is a SEPARATE, shorter list used only for grading/marking
guidance — it must never replace the full written answer.

CONTEXT:
{context}

Return JSON with this exact schema:
{{
  "insufficient_context": false,
  "questions": [
    {{
      "question": "string",
      "model_answer": "the FULL written answer (200-500 words, real prose, covering every part of the question)",
      "marking_points": ["key point 1", "key point 2", "key point 3"],
      "difficulty": "Easy" | "Medium" | "Hard",
      "bloom_level": "string",
      "source_chapter": "string",
      "source_page": "string"
    }}
  ]
}}
"""

# NOTE: no longer used by rag/generator.py -- Mixed Paper generation is now
# split into separate batched calls using the single-type templates below
# (MCQ_PROMPT_TEMPLATE, SHORT_QUESTION_PROMPT_TEMPLATE,
# LONG_QUESTION_PROMPT_TEMPLATE) to keep each individual request safely
# within Groq's per-minute token budget. Left here in case a future
# single-call use case wants it back.
MIXED_PAPER_PROMPT_TEMPLATE = GROUNDING_RULES + """
TASK: Generate a mixed examination paper section from the CONTEXT below:
- {num_mcq} MCQs
- {num_short} Short-Answer Questions
- {num_long} Long-Answer Questions
Target difficulty: "{difficulty}". Bloom's Taxonomy level(s): {bloom_levels}.

For each long question, write a COMPLETE, FULLY WRITTEN model answer in
flowing prose (not just fragments) that addresses every part of the
question. "marking_points" is a separate, shorter grading-guidance list —
it must never replace the full written answer.

CONTEXT:
{context}

Return JSON with this exact schema:
{{
  "insufficient_context": false,
  "mcqs": [
    {{
      "question": "string",
      "options": {{"A": "string", "B": "string", "C": "string", "D": "string"}},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "string",
      "difficulty": "string",
      "bloom_level": "string",
      "source_chapter": "string",
      "source_page": "string"
    }}
  ],
  "short_questions": [
    {{
      "question": "string",
      "model_answer": "string",
      "difficulty": "string",
      "bloom_level": "string",
      "source_chapter": "string",
      "source_page": "string"
    }}
  ],
  "long_questions": [
    {{
      "question": "string",
      "model_answer": "the FULL written answer (200-500 words, real prose, covering every part of the question)",
      "marking_points": ["string", "string"],
      "difficulty": "string",
      "bloom_level": "string",
      "source_chapter": "string",
      "source_page": "string"
    }}
  ]
}}
"""

ANSWER_KEY_PROMPT_TEMPLATE = GROUNDING_RULES + """
TASK: You are producing a separate answer key for an already-generated exam
paper. Use the CONTEXT to verify and, where needed, fill in or improve the
model answers and marking points supplied. Do not change the meaning of any
answer that is already correct and complete.

For "Long" type questions, the "answer" field must be a COMPLETE, fully
written model answer in real prose (not a terse list of fragments) that
addresses every part of the question — not just a summary phrase.

QUESTIONS (JSON):
{questions_json}

CONTEXT:
{context}

Return JSON with this exact schema:
{{
  "insufficient_context": false,
  "answer_key": [
    {{
      "question_number": 1,
      "type": "MCQ" | "Short" | "Long",
      "answer": "for MCQ: the correct option letter with a brief explanation; for Short: the model answer; for Long: the FULL written model answer in complete prose"
    }}
  ]
}}
"""


def build_topic_queries(
    subject: str, chapters: list, bloom_levels: list, n: int, min_per_chapter: int = 3
) -> list:
    """Generate a spread of representative (chapter, query_text) pairs to
    pull diverse context for question generation, since there's no single
    natural-language "question" to search with at generation time.

    Every chapter in `chapters` is guaranteed at least `min_per_chapter`
    queries pinned to it specifically (used by retrieve_diverse_sample to
    filter to that one chapter), so retrieval doesn't end up dominated by
    whichever chapter happens to score highest on generic semantic
    similarity — a real risk when chapters share vocabulary (e.g. an
    intro/basics unit vs. later units).

    If `chapters` is empty (the person chose "Entire Document"), pass the
    full list of chapters actually present in the document instead of an
    empty list — otherwise there's nothing to round-robin across.
    """
    base_terms = chapters if chapters else [subject]

    # Guarantee even coverage: at least min_per_chapter queries per
    # chapter, topped up with additional round-robin queries if the
    # requested question count needs more than that baseline.
    baseline_count = max(n, min_per_chapter * len(base_terms))

    pairs = []
    for i in range(baseline_count):
        chapter = base_terms[i % len(base_terms)]
        bloom = bloom_levels[i % len(bloom_levels)] if bloom_levels else "Understand"
        query = f"{chapter} key concepts, definitions, and examples ({bloom} level)"
        pairs.append((chapter, query))
    return pairs
