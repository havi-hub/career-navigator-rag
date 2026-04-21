# Career Navigator — AI-Powered CV-to-Job Match Engine

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.4-green?logo=chainlink)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai)
![Streamlit](https://img.shields.io/badge/Streamlit-1.56-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Career Navigator is a personalized RAG (Retrieval-Augmented Generation) system that semantically matches a candidate's resume against a curated job description database and delivers actionable, GPT-4o-powered career coaching — not just similarity scores. The result is a concrete skill gap analysis and a fully rewritten resume summary, tailored to the best-fit role.

---

## System Architecture

The pipeline follows a classic RAG pattern, with a Streamlit interface as the delivery layer:

```
┌─────────────────────┐
│  data/jobs.json     │  Job descriptions (title + description)
└────────┬────────────┘
         │ OpenAI Embeddings (text-embedding-ada-002)
         ▼
┌─────────────────────┐
│  FAISS Vector DB    │  Local vector index persisted to faiss_index/
└────────┬────────────┘
         │ Semantic similarity search (cosine)
         ▼
┌─────────────────────┐
│  Resume PDF         │  Uploaded via UI or CLI — text extracted with pypdf
└────────┬────────────┘
         │ Top-1 matched job description as context
         ▼
┌─────────────────────┐
│  GPT-4o Analysis    │  Structured output via Pydantic — no prompt fragility
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit UI       │  Matching skills · Skill gaps · Rewritten summary
└─────────────────────┘
```

**Key design decision:** The LLM response is parsed via `with_structured_output` and a Pydantic schema — returning typed fields (`matching_skills`, `missing_skills`, `rewritten_summary`) rather than fragile string matching on section headers.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Package management | [`uv`](https://docs.astral.sh/uv/) — 10–100x faster than pip |
| LLM orchestration | LangChain (`langchain-openai`, `langchain-community`) |
| Embeddings | OpenAI `text-embedding-ada-002` |
| Vector store | FAISS (local, no infra required) |
| LLM | OpenAI GPT-4o with structured output |
| PDF parsing | pypdf |
| Web UI | Streamlit |
| Config | python-dotenv |

---

## Local Setup & Installation

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/getting-started/installation/), an OpenAI API key.

```bash
# 1. Clone the repository
git clone https://github.com/your-username/career-navigator.git
cd career-navigator

# 2. Install all dependencies (uv creates the virtual environment automatically)
uv sync

# 3. Configure environment variables
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Build the FAISS vector index from the job descriptions
uv run python build_vector_db.py

# 5. Launch the web app
uv run streamlit run app.py
```

Open `http://localhost:8501`, upload a resume PDF, and click **Analyze Resume**.

**CLI alternative (no UI):**
```bash
uv run python analyze_cv.py                        # uses default resume
uv run python analyze_cv.py path/to/resume.pdf     # custom resume path
```

**Adding new job descriptions:** Edit `data/jobs.json` (add a `title` and `description` per entry), then re-run `build_vector_db.py` to rebuild the index.

---

## Key Features & Business Impact

| Feature | What it means in practice |
|---|---|
| **Semantic job matching** | Finds the most relevant role using vector similarity, not keyword overlap — works even when resume and job use different terminology |
| **Skill gap analysis** | Surfaces the top 3 missing skills so a candidate knows exactly what to address before applying |
| **AI resume rewriting** | Generates a role-specific professional summary — ready to paste, not just feedback to act on |
| **Structured LLM output** | Pydantic-validated responses ensure consistent, parseable results regardless of model verbosity |
| **Extensible job catalog** | Job descriptions live in `data/jobs.json` — no code changes required to add or update roles |
| **Zero-infra vector search** | FAISS runs entirely locally; no cloud vector DB needed for development or demos |
