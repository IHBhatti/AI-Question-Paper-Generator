"""
app.py
------
Streamlit front-end for the AI Test Paper Generator.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import streamlit as st

from config import (
    BLOOM_LEVELS,
    DEFAULT_MARKS,
    DIFFICULTY_LEVELS,
    LOG_FILE,
    LOG_LEVEL,
    QUESTION_TYPES,
    UPLOADS_DIR,
)
from rag.chunker import chunk_document, get_available_chapters
from rag.embeddings import (
    build_vectorstore,
    is_duplicate_document,
    load_chunks_cache,
    load_vectorstore,
    register_document_hash,
    save_chunks_cache,
    save_vectorstore,
)
from rag.generator import (
    InsufficientContextError,
    RateLimitedError,
    TruncatedResponseError,
    generate_answer_key,
    generate_questions,
    merge_answer_key,
)
from rag.loader import compute_file_hash, load_document
from rag.parser import build_manual_chapters, detect_chapters
from utils.pdf_export import export_answer_key_pdf, export_test_paper_pdf
from utils.text_export import export_answer_key_txt, export_test_paper_txt

# --------------------------------------------------------------------------
# Logging setup
# --------------------------------------------------------------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("app")

st.set_page_config(
    page_title="AI Test Paper Generator",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Visual theme (UI only — purely cosmetic CSS, no behavior is affected)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }

    /* ---- App background ---- */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #f5f7ff 0%, #ffffff 40%, #fef6ff 100%);
    }

    /* ---- Sidebar ---- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4b3f72 0%, #6a4c93 55%, #a15fa8 100%);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small {
        color: #f5f0ff !important;
    }

    /* ---- Hero banner for the main title ---- */
    .hero-banner {
        padding: 1.6rem 2rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #6a4c93 0%, #a15fa8 50%, #ff8fab 100%);
        box-shadow: 0 10px 30px rgba(106, 76, 147, 0.25);
        margin-bottom: 1.4rem;
    }
    .hero-banner h1 {
        color: #ffffff !important;
        margin: 0;
        font-weight: 700;
        font-size: 2.1rem;
    }
    .hero-banner p {
        color: #f3e9ff;
        margin: 0.35rem 0 0 0;
        font-size: 1.02rem;
    }

    /* ---- Section headers ---- */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 1rem;
        border-radius: 12px;
        background: linear-gradient(90deg, #eef0ff 0%, #fdeeff 100%);
        border-left: 5px solid #8a5cf6;
        margin: 1.1rem 0 0.8rem 0;
    }
    .section-header .badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px; height: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #8a5cf6, #ff8fab);
        color: white; font-weight: 700; font-size: 0.85rem;
        flex-shrink: 0;
    }
    .section-header span.title-text {
        font-weight: 600;
        font-size: 1.08rem;
        color: #3d2a5c;
    }

    /* ---- Buttons ---- */
    .stButton>button, .stDownloadButton>button {
        border-radius: 10px;
        border: none;
        font-weight: 600;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton>button[kind="primary"], .stDownloadButton>button {
        background: linear-gradient(120deg, #8a5cf6, #ff8fab);
        color: white;
    }
    .stButton>button[kind="primary"]:hover, .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(138, 92, 246, 0.35);
    }
    .stButton>button[kind="secondary"] {
        background: #ffffff;
        border: 1.5px solid #d9c8ff !important;
        color: #6a4c93;
    }

    /* ---- Alerts / status boxes ---- */
    [data-testid="stAlertContentSuccess"] { color: #14532d; }
    div[data-baseweb="notification"] { border-radius: 12px !important; }

    /* ---- Progress bar ---- */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #8a5cf6, #ff8fab);
    }

    /* ---- Metrics / caption chips ---- */
    .chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: linear-gradient(120deg, #eef0ff, #ffe6f2);
        border: 1px solid #e2d4ff;
        color: #5b3a8e;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.15rem 0.25rem 0.15rem 0;
    }

    /* ---- Dividers ---- */
    hr { border-top: 2px solid #eadcff; }

    /* ---- Tabs / expander / status widget ---- */
    [data-testid="stExpander"], .streamlit-expanderHeader {
        border-radius: 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def section_header(icon: str, text: str) -> None:
    """Colorful pill-style section header (visual only — replaces plain
    st.subheader calls, does not affect any state or logic)."""
    st.markdown(
        f"""<div class="section-header"><span class="badge">{icon}</span>
        <span class="title-text">{text}</span></div>""",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------
DEFAULT_STATE = {
    "vectorstore": None,
    "chunks": None,
    "chapters": [],
    "chapter_names": [],
    "doc_loaded": False,
    "source_filename": None,
    "history": [],
    "last_paper": None,
    "last_meta": None,
}
for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Sidebar: document upload + chapter detection
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Document Setup")

    uploaded_file = st.file_uploader(
        "Upload study material (PDF or TXT)", type=["pdf", "txt"]
    )

    if uploaded_file is not None:
        process_clicked = st.button("Process Document", type="primary", use_container_width=True)
        if process_clicked:
            save_path = UPLOADS_DIR / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.status("Processing document...", expanded=True) as status:
                try:
                    file_hash = compute_file_hash(save_path)
                    cached_vectorstore = load_vectorstore(file_hash)
                    cached_chunks = load_chunks_cache(file_hash) if cached_vectorstore else None

                    if cached_vectorstore is not None and cached_chunks is not None:
                        st.write("Found a previously processed version of this exact file — loading instantly (no re-OCR/re-embedding needed)...")
                        st.session_state["vectorstore"] = cached_vectorstore
                        st.session_state["chunks"] = cached_chunks
                        st.session_state["chapters"] = []
                        st.session_state["chapter_names"] = get_available_chapters(cached_chunks)
                        st.session_state["doc_loaded"] = True
                        st.session_state["source_filename"] = uploaded_file.name
                        status.update(label="Document ready ✅ (loaded from cache)", state="complete")
                    else:
                        if is_duplicate_document(file_hash):
                            st.warning(
                                "⚠️ This exact document has already been processed before. "
                                "Re-processing anyway."
                            )

                        st.write("Extracting text...")
                        ocr_progress_bar = st.progress(0.0, text="Checking pages...")

                        def _ocr_progress(done: int, total: int) -> None:
                            ocr_progress_bar.progress(
                                done / total, text=f"Running OCR on scanned pages: {done}/{total}"
                            )

                        doc = load_document(
                            save_path,
                            source_filename=uploaded_file.name,
                            ocr_progress_callback=_ocr_progress,
                        )
                        ocr_progress_bar.empty()

                        if doc.is_scanned_document:
                            st.info(
                                f"📷 This looks like a scanned document ({len(doc.ocr_page_numbers)} "
                                f"of {len(doc.pages)} pages had no text layer) — OCR was used to "
                                "extract the content. This is cached, so re-uploading the same "
                                "file again will be instant."
                            )

                        st.write("Detecting chapters...")
                        chapters = detect_chapters(doc)

                        # Always route through a review step rather than
                        # committing immediately: OCR'd chapter titles can
                        # come out noisy (e.g. a misread word or two), so
                        # the user gets a chance to fix names or page
                        # numbers before the (slow) embedding step runs.
                        st.session_state["_pending_doc"] = doc
                        st.session_state["_pending_chapters"] = chapters
                        if chapters:
                            status.update(
                                label=f"Detected {len(chapters)} chapter(s) — review below ⬇️",
                                state="complete",
                            )
                        else:
                            status.update(
                                label="Could not auto-detect chapters — define them manually below",
                                state="error",
                            )

                except Exception as e:
                    logger.exception("Failed to process document")
                    status.update(label="Processing failed", state="error")
                    st.error(f"Error processing document: {e}")

    # ----------------------------------------------------------------------
    # Chapter review / manual editing step
    #
    # Shown whenever a document has been loaded but not yet confirmed —
    # whether chapters were auto-detected (pre-filled, editable) or need to
    # be entered from scratch (auto-detection failed).
    # ----------------------------------------------------------------------
    if st.session_state.get("_pending_doc") is not None and not st.session_state["doc_loaded"]:
        pending_chapters = st.session_state.get("_pending_chapters") or []

        if pending_chapters:
            st.subheader("Review Detected Chapters")
            st.caption(
                "Auto-detection can occasionally misread a title from a scanned page. "
                "Fix any names or starting pages below, then confirm."
            )
            editable_rows = [
                {"Chapter Name": c.name, "Start Page": c.start_page} for c in pending_chapters
            ]
            edited = st.data_editor(
                editable_rows,
                num_rows="dynamic",
                use_container_width=True,
                key="chapter_review_editor",
                column_config={
                    "Chapter Name": st.column_config.TextColumn(required=True),
                    "Start Page": st.column_config.NumberColumn(required=True, min_value=1, step=1),
                },
            )
            confirm_label = "Confirm Chapters & Build Index"
        else:
            st.subheader("Manual Chapter Definition")
            st.caption(
                "Could not auto-detect chapters. Enter each chapter's name and starting page below."
            )
            edited = st.data_editor(
                [{"Chapter Name": "", "Start Page": 1}],
                num_rows="dynamic",
                use_container_width=True,
                key="chapter_manual_editor",
                column_config={
                    "Chapter Name": st.column_config.TextColumn(required=True),
                    "Start Page": st.column_config.NumberColumn(required=True, min_value=1, step=1),
                },
            )
            confirm_label = "Apply Chapters & Build Index"

        if st.button(confirm_label, type="primary", use_container_width=True):
            try:
                doc = st.session_state["_pending_doc"]
                rows = sorted(
                    [r for r in edited if r.get("Chapter Name", "").strip()],
                    key=lambda r: r["Start Page"],
                )
                if not rows:
                    st.error("Add at least one chapter with a name and starting page.")
                    st.stop()

                page_ranges = {}
                total_pages = len(doc.pages)
                for i, row in enumerate(rows):
                    start = int(row["Start Page"])
                    end = int(rows[i + 1]["Start Page"]) - 1 if i + 1 < len(rows) else total_pages
                    page_ranges[row["Chapter Name"].strip()] = (start, max(start, end))

                chapters = build_manual_chapters(page_ranges)
                chunks = chunk_document(doc, chapters)
                vectorstore = build_vectorstore(chunks)
                register_document_hash(doc.file_hash)
                save_vectorstore(vectorstore, doc.file_hash)
                save_chunks_cache(chunks, doc.file_hash)

                st.session_state["vectorstore"] = vectorstore
                st.session_state["chunks"] = chunks
                st.session_state["chapters"] = chapters
                st.session_state["chapter_names"] = get_available_chapters(chunks)
                st.session_state["doc_loaded"] = True
                st.session_state["source_filename"] = doc.source_filename
                st.session_state["_pending_doc"] = None
                st.session_state["_pending_chapters"] = None
                st.success("Chapters confirmed. Document is ready.")
                st.rerun()
            except Exception as e:
                logger.exception("Failed to apply chapters")
                st.error(f"Could not apply chapters: {e}")

    if st.session_state["doc_loaded"]:
        st.success(f"✅ Loaded: {st.session_state['source_filename']}")
        st.markdown(
            f"""<span class="chip">📦 {len(st.session_state['chunks'])} chunks</span>"""
            f"""<span class="chip">📚 {len(st.session_state['chapter_names'])} chapter(s)</span>""",
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("🕘 Session History")
    if st.session_state["history"]:
        for item in reversed(st.session_state["history"][-10:]):
            st.caption(f"• {item}")
    else:
        st.caption("No papers generated yet this session.")


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>📝 AI Test Paper Generator</h1>
        <p>Generate professional exam papers strictly from your uploaded study material
        using Retrieval-Augmented Generation — no hallucinated content.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["doc_loaded"]:
    st.info("👈 Upload and process a PDF or TXT document from the sidebar to get started.")
    st.stop()

section_header("1", "Chapter Selection")
chapter_names = st.session_state["chapter_names"]

selection_mode = st.radio(
    "Coverage",
    ["Entire Document", "Select Chapters"],
    horizontal=True,
)

selected_chapters = []
if selection_mode == "Select Chapters":
    cols = st.columns(3)
    for i, name in enumerate(chapter_names):
        with cols[i % 3]:
            if st.checkbox(name, key=f"chap_{name}"):
                selected_chapters.append(name)
    if not selected_chapters:
        st.warning("Select at least one chapter, or switch to Entire Document.")

section_header("2", "Question Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    question_type = st.selectbox("Question Type", QUESTION_TYPES)
with col2:
    difficulty = st.selectbox("Difficulty", DIFFICULTY_LEVELS)
with col3:
    num_questions = st.number_input("Number of Questions", min_value=1, max_value=60, value=20)

if question_type in ("Long Questions", "Mixed Paper") and num_questions > 20:
    st.caption(
        "ℹ️ Long questions are generated a couple at a time (each needs a full written "
        "answer), so a larger count here means more, slower AI requests behind the scenes."
    )

bloom_levels = st.multiselect("Bloom's Taxonomy Levels (optional)", BLOOM_LEVELS)

section_header("3", "Marks & Timing")
col4, col5, col6 = st.columns(3)
with col4:
    marks_mcq = st.number_input("Marks per MCQ", min_value=1, value=DEFAULT_MARKS["MCQ"])
with col5:
    marks_short = st.number_input("Marks per Short Question", min_value=1, value=DEFAULT_MARKS["Short"])
with col6:
    marks_long = st.number_input("Marks per Long Question", min_value=1, value=DEFAULT_MARKS["Long"])

col7, col8, col9 = st.columns(3)
with col7:
    time_duration = st.selectbox("Time Duration", ["60 Minutes", "90 Minutes", "120 Minutes", "Custom"])
    if time_duration == "Custom":
        time_duration = st.text_input("Custom duration", value="60 Minutes")
with col8:
    exam_date = st.date_input("Exam Date", value=date.today())
with col9:
    subject_name = st.text_input("Subject Name", value="")

col10, col11 = st.columns(2)
with col10:
    institution_name = st.text_input("Institution Name (optional)", value="")
with col11:
    st.write("")

instructions = st.text_area(
    "Instructions (optional)",
    value="Attempt all questions.\nRead carefully.\nNo calculators allowed.",
    height=90,
)

generate_clicked = st.button("🚀 Generate Test Paper", type="primary", use_container_width=True)

if generate_clicked:
    if selection_mode == "Select Chapters" and not selected_chapters:
        st.error("Please select at least one chapter.")
        st.stop()
    if not subject_name.strip():
        st.error("Please enter a subject name.")
        st.stop()

    chapters_for_query = selected_chapters if selection_mode == "Select Chapters" else []

    mixed_split = None
    if question_type == "Mixed Paper":
        mixed_split = {
            "mcq": int(num_questions * 0.65),
            "short": int(num_questions * 0.25),
            "long": max(1, num_questions - int(num_questions * 0.65) - int(num_questions * 0.25)),
        }

    with st.spinner("Retrieving relevant content and generating questions..."):
        try:
            result = generate_questions(
                vectorstore=st.session_state["vectorstore"],
                subject=subject_name,
                chapters=chapters_for_query,
                question_type=question_type,
                difficulty=difficulty,
                bloom_levels=bloom_levels,
                num_questions=int(num_questions),
                mixed_split=mixed_split,
                all_available_chapters=st.session_state.get("chapter_names"),
            )
        except InsufficientContextError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except TruncatedResponseError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except RateLimitedError as e:
            st.error(f"⏳ {e}")
            st.stop()
        except RuntimeError as e:
            st.error(f"⚠️ {e}")
            st.stop()
        except Exception as e:
            logger.exception("Generation failed")
            st.error(f"Generation failed: {e}")
            st.stop()

    # Normalize result shape across question types
    if question_type == "Mixed Paper":
        mcqs = result.get("mcqs", [])
        short_qs = result.get("short_questions", [])
        long_qs = result.get("long_questions", [])
    elif question_type == "MCQs":
        mcqs, short_qs, long_qs = result.get("questions", []), [], []
    elif question_type == "Short Questions":
        mcqs, short_qs, long_qs = [], result.get("questions", []), []
    else:  # Long Questions
        mcqs, short_qs, long_qs = [], [], result.get("questions", [])

    total_marks = len(mcqs) * marks_mcq + len(short_qs) * marks_short + len(long_qs) * marks_long

    meta = {
        "institution_name": institution_name,
        "subject": subject_name,
        "exam_date": exam_date.strftime("%d %B %Y"),
        "time_duration": time_duration,
        "total_marks": total_marks,
        "chapters": chapters_for_query,
        "instructions": instructions,
    }
    marks = {"MCQ": marks_mcq, "Short": marks_short, "Long": marks_long}

    st.session_state["last_paper"] = {"mcqs": mcqs, "short": short_qs, "long": long_qs}
    st.session_state["last_meta"] = meta
    st.session_state["last_marks"] = marks
    # A fresh paper invalidates any cached answer-key data and previously
    # generated download files from an earlier paper in this session.
    for stale_key in (
        "_merged_answer_data",
        "_paper_pdf_path",
        "_paper_txt_path",
        "_key_pdf_path",
        "_key_txt_path",
    ):
        st.session_state.pop(stale_key, None)
    st.session_state["history"].append(
        f"{subject_name} — {question_type} ({num_questions}q) — {exam_date}"
    )
    st.success("✅ Test paper generated successfully!")

# --------------------------------------------------------------------------
# Render generated paper + downloads
# --------------------------------------------------------------------------
if st.session_state.get("last_paper"):
    paper = st.session_state["last_paper"]
    meta = st.session_state["last_meta"]
    marks = st.session_state["last_marks"]

    st.divider()
    section_header("📄", "Generated Test Paper Preview")

    if meta.get("institution_name"):
        st.markdown(f"### {meta['institution_name']}")
    st.markdown(
        f"""<span class="chip">📘 {meta['subject']}</span>"""
        f"""<span class="chip">📅 {meta['exam_date']}</span>"""
        f"""<span class="chip">⏱️ {meta['time_duration']}</span>"""
        f"""<span class="chip">🏆 {meta['total_marks']} marks</span>""",
        unsafe_allow_html=True,
    )
    st.markdown(f"**Chapters Covered:** {', '.join(meta['chapters']) or 'Entire Document'}")

    if paper["mcqs"]:
        st.markdown("#### Section A — Multiple Choice Questions")
        for i, q in enumerate(paper["mcqs"], start=1):
            st.markdown(f"**{i}. {q['question']}**")
            for letter, opt in q.get("options", {}).items():
                st.markdown(f"&nbsp;&nbsp;({letter}) {opt}")

    if paper["short"]:
        st.markdown("#### Section B — Short Questions")
        for i, q in enumerate(paper["short"], start=1):
            st.markdown(f"**{i}. {q['question']}**")

    if paper["long"]:
        st.markdown("#### Section C — Long Questions")
        for i, q in enumerate(paper["long"], start=1):
            st.markdown(f"**{i}. {q['question']}**")

    st.divider()
    section_header("⬇️", "Downloads")

    dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)

    safe_subject = "".join(c if c.isalnum() else "_" for c in meta["subject"])[:40] or "paper"

    with dl_col1:
        if st.button("Generate Paper PDF", use_container_width=True):
            path = export_test_paper_pdf(
                meta, paper["mcqs"], paper["short"], paper["long"], marks,
                output_filename=f"{safe_subject}_test_paper.pdf",
            )
            st.session_state["_paper_pdf_path"] = path
        if st.session_state.get("_paper_pdf_path"):
            p = Path(st.session_state["_paper_pdf_path"])
            st.download_button("Download Paper PDF", data=p.read_bytes(), file_name=p.name, mime="application/pdf", use_container_width=True)

    with dl_col2:
        if st.button("Generate Paper TXT", use_container_width=True):
            path = export_test_paper_txt(
                meta, paper["mcqs"], paper["short"], paper["long"], marks,
                output_filename=f"{safe_subject}_test_paper.txt",
            )
            st.session_state["_paper_txt_path"] = path
        if st.session_state.get("_paper_txt_path"):
            p = Path(st.session_state["_paper_txt_path"])
            st.download_button("Download Paper TXT", data=p.read_bytes(), file_name=p.name, mime="text/plain", use_container_width=True)

    with dl_col3:
        if st.button("Generate Answer Key PDF", use_container_width=True):
            with st.spinner("Verifying answers against source material..."):
                try:
                    key_result = generate_answer_key(
                        st.session_state["vectorstore"],
                        {"mcqs": paper["mcqs"], "short_questions": paper["short"], "long_questions": paper["long"]},
                        meta["chapters"],
                    )
                except Exception:
                    key_result = None

            mcqs_final, short_final, long_final = merge_answer_key(
                paper["mcqs"], paper["short"], paper["long"], key_result
            )
            # Cache the merged (verified + gap-filled) answer data so the
            # TXT button below can reuse it without a second LLM call.
            st.session_state["_merged_answer_data"] = (mcqs_final, short_final, long_final)

            missing_answers = sum(1 for q in long_final if not q.get("model_answer"))
            if missing_answers:
                st.warning(
                    f"⚠️ {missing_answers} long question(s) still have no full written answer "
                    "even after verification — the source material may not cover them well enough. "
                    "Consider selecting more chapters or reviewing those questions manually."
                )

            path = export_answer_key_pdf(
                meta, mcqs_final, short_final, long_final,
                output_filename=f"{safe_subject}_answer_key.pdf",
            )
            st.session_state["_key_pdf_path"] = path
        if st.session_state.get("_key_pdf_path"):
            p = Path(st.session_state["_key_pdf_path"])
            st.download_button("Download Answer Key PDF", data=p.read_bytes(), file_name=p.name, mime="application/pdf", use_container_width=True)

    with dl_col4:
        if st.button("Generate Answer Key TXT", use_container_width=True):
            cached = st.session_state.get("_merged_answer_data")
            if cached is not None:
                mcqs_final, short_final, long_final = cached
            else:
                with st.spinner("Verifying answers against source material..."):
                    try:
                        key_result = generate_answer_key(
                            st.session_state["vectorstore"],
                            {"mcqs": paper["mcqs"], "short_questions": paper["short"], "long_questions": paper["long"]},
                            meta["chapters"],
                        )
                    except Exception:
                        key_result = None
                mcqs_final, short_final, long_final = merge_answer_key(
                    paper["mcqs"], paper["short"], paper["long"], key_result
                )
                st.session_state["_merged_answer_data"] = (mcqs_final, short_final, long_final)

            path = export_answer_key_txt(
                meta, mcqs_final, short_final, long_final,
                output_filename=f"{safe_subject}_answer_key.txt",
            )
            st.session_state["_key_txt_path"] = path
        if st.session_state.get("_key_txt_path"):
            p = Path(st.session_state["_key_txt_path"])
            st.download_button("Download Answer Key TXT", data=p.read_bytes(), file_name=p.name, mime="text/plain", use_container_width=True)
