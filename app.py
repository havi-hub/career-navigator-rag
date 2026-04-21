import tempfile
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pypdf import PdfReader
from pydantic import BaseModel

load_dotenv()

st.set_page_config(page_title="Career Navigator", page_icon="🧭", layout="centered")
st.title("🧭 Career Navigator")
st.caption("Upload your resume and get instant AI-powered career gap analysis.")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type="pdf")

analyze_clicked = st.button("Analyze Resume", type="primary", disabled=uploaded_file is None)

if analyze_clicked and uploaded_file:
    with st.spinner("Analyzing your resume..."):
        # Extract text from uploaded PDF
        reader = PdfReader(uploaded_file)
        resume_text = "".join(page.extract_text() or "" for page in reader.pages)

        if not resume_text.strip():
            st.error("Could not extract text from the PDF. Please try a different file.")
            st.stop()

        # Load FAISS index
        index_path = "faiss_index"
        if not os.path.exists(index_path):
            st.error("Job index not found. Please run `python build_vector_db.py` first.")
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

    # Results
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
