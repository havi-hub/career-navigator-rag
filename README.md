# Career Navigator

Career Navigator is a resume-to-role intelligence system with two retrieval paths:

- **Static RAG path** for deterministic matching against curated local job data.
- **Live market path** for near-real-time role discovery and ranking from Indeed Israel.

Both paths use structured LLM outputs and semantic retrieval to produce actionable gap analysis for a candidate profile.

## Ethical Disclaimer

This project is for **educational and portfolio purposes only**.

The web scraping capability in `live_jobs.py` is implemented exclusively to demonstrate data engineering, retrieval, and LLM orchestration techniques in a local development environment. Scraped content is processed in-memory at runtime for analysis and is not persisted for commercial use, redistribution, or resale.

Users are solely responsible for operating this code in compliance with applicable laws, platform Terms of Service, and rate/access restrictions.

## Architecture Overview

### 1) Static RAG Pipeline (`Analyze Resume`)

1. `build_vector_db.py` loads `data/jobs.json` and builds a local FAISS index.
2. `app.py` (or `analyze_cv.py`) extracts resume text from PDF via `pypdf`.
3. Resume text is embedded and queried against FAISS (`k=1`).
4. Matched job context is analyzed by `gpt-4o` with Pydantic structured output.
5. UI renders matching skills, missing skills, and a rewritten summary.

### 2) Live Job Search + Re-Ranked RAG (`Search Live Indeed Jobs`)

`run_live_job_search()` in `live_jobs.py` implements a multi-stage pipeline:

1. **CV profile distillation** (`gpt-4o-mini`) into a compact technical query profile.
2. **Playwright-based job discovery** on `il.indeed.com` (async browser automation).
3. **Concurrent page scraping** with request blocking for non-essential resources (images/media/fonts/stylesheets).
4. **Language normalization layer**: each description is translated to English when Hebrew or mixed text is detected (`gpt-4o-mini`).
5. **Suitability classification** (`gpt-4o-mini`) to filter out analyst/BI/junior-heavy postings based on required day-to-day skills.
6. **In-memory FAISS build** over qualified live jobs.
7. **Semantic retrieval** of top candidates (`k <= 20`) using the distilled CV profile.
8. **Scoring and re-ranking**: each retrieved candidate gets a Scientific Rigor score (1-10) from `gpt-4o-mini`.
9. **Deep analysis stage** (`gpt-4o`): only top 3 scored jobs receive detailed fit/gap analysis.
10. UI presents ranked results with rationale, fit summary, matching skills, and missing skills.

This staged design keeps expensive model usage focused on high-confidence candidates while preserving broad recall from retrieval.

## Key Architectural Changes Implemented

- Migrated live search from simple request-driven flow to **async Playwright orchestration** with stealth and selective resource loading.
- Added a **translation normalization layer** so multilingual job descriptions can be compared consistently in the same semantic space.
- Introduced **post-retrieval scoring/re-ranking** with `gpt-4o-mini` (Scientific Rigor rubric) before expensive analysis.
- Added **top-k deep analysis gating**: only the highest-ranked jobs are passed to `gpt-4o` for full gap analysis.
- Preserved **structured outputs** via Pydantic models across classification, scoring, and analysis stages.

## Repository Components

- `app.py`: Streamlit UI with two actions: static index analysis and live job search.
- `live_jobs.py`: async scraping, translation, filtering, retrieval, scoring, and deep-analysis orchestration.
- `build_vector_db.py`: offline index build from `data/jobs.json`.
- `analyze_cv.py`: CLI version of static RAG analysis.
- `test_connection.py`: environment/API connectivity check.

## Technology Stack

- Python 3.12
- Streamlit
- LangChain (`langchain-openai`, `langchain-community`)
- OpenAI models: `gpt-4o-mini` (triage/scoring/translation), `gpt-4o` (deep analysis)
- OpenAI embeddings + FAISS
- Playwright + `playwright-stealth`
- BeautifulSoup4
- `pypdf`
- `python-dotenv`
- `uv` for dependency and environment management

## Setup

Prerequisites:

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- `OPENAI_API_KEY`

Install:

```bash
uv sync
```

Environment:

```bash
OPENAI_API_KEY=sk-...
```

## Run

Build static index (required for static RAG flow):

```bash
uv run python build_vector_db.py
```

Launch app:

```bash
uv run streamlit run app.py
```

CLI static analysis:

```bash
uv run python analyze_cv.py
uv run python analyze_cv.py path/to/resume.pdf
```

Connectivity check:

```bash
uv run python test_connection.py
```

## Operational Notes

- Live search currently targets `il.indeed.com` with query/location defaults defined in `live_jobs.py`.
- Scraping quality depends on website availability and anti-automation behavior.
- The static path requires a prebuilt `faiss_index/`.
- `data/` and `faiss_index/` are intentionally ignored in version control.
