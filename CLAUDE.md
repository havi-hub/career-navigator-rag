# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Disclaimer

This project is for **educational and portfolio purposes only**.

The web scraping implementation in `live_jobs.py` exists solely to demonstrate data engineering and LLM integration patterns in local development. Scraped job postings are processed for runtime analysis and are not intended for commercial redistribution, resale, or persistent dataset creation.

All scraping-related work must remain compliant with platform Terms of Service and applicable legal requirements.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
uv sync                                        # Install all dependencies
uv run python build_vector_db.py               # Build static FAISS index from data/jobs.json
uv run python analyze_cv.py                    # Run static RAG analysis on default resume
uv run python analyze_cv.py path/to/resume.pdf # Run static RAG analysis on a specific resume
uv run streamlit run app.py                    # Launch the Streamlit web UI
uv run python test_connection.py               # Verify OpenAI API key is valid and reachable
```

## Architecture

The system has two analysis paths exposed by `app.py`.

### 1) Static RAG path (`Analyze Resume`)

- `build_vector_db.py` builds a persistent FAISS index from `data/jobs.json`.
- `analyze_cv.py` (and the corresponding `app.py` branch) extracts PDF text, retrieves the best match from FAISS, and runs `gpt-4o` with structured output for:
  - `matching_skills`
  - `missing_skills`
  - `rewritten_summary`

### 2) Live search + re-ranked RAG path (`Search Live Indeed Jobs`)

Implemented in `live_jobs.py` through `run_live_job_search()`:

1. Distill candidate profile from resume using `gpt-4o-mini`.
2. Discover and scrape live postings from `il.indeed.com` with async Playwright.
3. Apply stealth browser hardening and block non-critical resources for speed.
4. Normalize multilingual descriptions via translation-to-English (`gpt-4o-mini`).
5. Classify and filter unsuitable jobs (analyst/BI/junior-heavy) using structured output.
6. Build in-memory FAISS over qualified jobs.
7. Retrieve top semantic candidates.
8. Score each candidate with a 1-10 Scientific Rigor rubric (`gpt-4o-mini`).
9. Re-rank and keep top 3.
10. Run deep fit/gap analysis on top 3 only with `gpt-4o`.

### Model allocation strategy

- `gpt-4o-mini`: profile extraction, translation, role suitability filtering, scoring/re-ranking.
- `gpt-4o`: deep final analysis on top-ranked results only.

This keeps cost and latency controlled while preserving high-quality reasoning where it matters.

## Adding Job Descriptions

Edit `data/jobs.json` with objects that include:

- `"title"`
- `"description"`

Then re-run `uv run python build_vector_db.py`.

## Environment

A `.env` file at the project root must contain:

```
OPENAI_API_KEY=sk-...
```

Notes:

- `analyze_cv.py` and the static `app.py` path require `faiss_index/` to exist.
- Live search does not require the static index, but it does require network access and a valid OpenAI API key.
