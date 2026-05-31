import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pypdf import PdfReader
from pydantic import BaseModel

from live_jobs import run_live_job_search

load_dotenv()

st.set_page_config(page_title="Career Navigator", page_icon="🧭", layout="centered")
st.title("🧭 Career Navigator")
st.caption("Upload your resume and get instant AI-powered career gap analysis.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    analyze_clicked = st.button("Analyze Resume", type="primary", disabled=uploaded_file is None)
with col_btn2:
    live_search_clicked = st.button("Search Live ATS Jobs", disabled=uploaded_file is None)

# ── Helper to extract resume text ──────────────────────────────────────────────
def extract_resume_text(file) -> str:
    reader = PdfReader(file)
    return "".join(page.extract_text() or "" for page in reader.pages)


# ── Analyze Resume (existing FAISS index) ─────────────────────────────────────
if analyze_clicked and uploaded_file:
    with st.spinner("Analyzing your resume..."):
        resume_text = extract_resume_text(uploaded_file)

        if not resume_text.strip():
            st.error("Could not extract text from the PDF. Please try a different file.")
            st.stop()

        index_path = "faiss_index"
        if not os.path.exists(index_path):
            st.error("Job index not found. Please run `uv run python build_vector_db.py` first.")
            st.stop()

        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(
            index_path, embeddings, allow_dangerous_deserialization=True
        )

        results = vector_store.similarity_search(resume_text, k=1)
        if not results:
            st.error("No matching job descriptions found in the index.")
            st.stop()

        matched_job = results[0]

        class CareerAnalysis(BaseModel):
            matching_skills: list[str]
            missing_skills: list[str]
            rewritten_summary: str

        llm = ChatOpenAI(model="gpt-4o").with_structured_output(CareerAnalysis)

        prompt = f"""You are an expert career coach and resume analyst.

Resume:
---
{resume_text}
---

Job Description:
---
{matched_job.page_content}
---

Return exactly:
- matching_skills: top 3 skills the resume shares with the job
- missing_skills: top 3 skills or gaps missing from the resume
- rewritten_summary: a professional summary paragraph tailored to this job
"""

        analysis: CareerAnalysis = llm.invoke(prompt)

    job_title = matched_job.metadata.get("title", "Unknown")
    st.success(f"Best matching role: **{job_title}**")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Matching Skills")
        for skill in analysis.matching_skills:
            st.markdown(f"- {skill}")
    with col2:
        st.subheader("⚠️ Skill Gaps")
        for skill in analysis.missing_skills:
            st.markdown(f"- {skill}")

    st.subheader("📝 Rewritten Summary")
    st.info(analysis.rewritten_summary)


# ── Search Live ATS Jobs ─────────────────────────────────────────────────────
if live_search_clicked and uploaded_file:
    resume_text = extract_resume_text(uploaded_file)

    if not resume_text.strip():
        st.error("Could not extract text from the PDF. Please try a different file.")
        st.stop()

    st.subheader("🌐 Live ATS Job Search")

    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(step: int, total: int, message: str):
        progress_bar.progress(step / total)
        status_text.text(message)

    try:
        live_results, cv_profile = run_live_job_search(resume_text, progress_callback=update_progress)
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Live search failed: {e}")
        st.stop()

    progress_bar.empty()
    status_text.empty()

    if not live_results:
        st.warning("No live job results were returned. Try again later.")
        st.stop()

    with st.expander("Profile used for matching", expanded=False):
        st.caption(
            "GPT-4o Mini distilled your CV into this focused technical profile before searching. "
            "Jobs were extracted from seed ATS boards via the local ATS MCP server, filtered with "
            "GPT-4o Mini, and the top 3 "
            "were deeply analyzed with GPT-4o."
        )
        st.markdown(cv_profile)

    st.success(f"Found and analyzed the top {len(live_results)} matching live jobs for your CV.")

    for i, job in enumerate(live_results, start=1):
        score = job.get("score", "?")
        expander_label = f"#{i} — {job['title']}  |  Scientific Rigor: {score}/10"
        with st.expander(expander_label, expanded=(i == 1)):
            if job["url"]:
                st.markdown(f"[View job posting]({job['url']})")

            score_color = "green" if score != "?" and score >= 8 else ("orange" if score != "?" and score >= 5 else "red")
            st.markdown(
                f"**Scientific Rigor Score: :{score_color}[{score}/10]**  \n"
                f"*{job.get('score_rationale', '')}*"
            )

            st.markdown("**Why it's a fit**")
            st.info(job["summary"])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**✅ Matching Skills**")
                for skill in job["matching_skills"]:
                    st.markdown(f"- {skill}")
            with col2:
                st.markdown("**⚠️ Missing Skills**")
                for skill in job["missing_skills"]:
                    st.markdown(f"- {skill}")
