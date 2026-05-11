# Career Navigator

Career Navigator is a resume-to-role intelligence system with two retrieval paths:

- **Static RAG path** — deterministic matching against a curated local job index.
- **Live ATS path** — near-real-time role discovery from Greenhouse boards, with multi-stage LLM filtering and semantic re-ranking.

Both paths use structured LLM outputs (Pydantic) and FAISS-based semantic retrieval to produce actionable fit/gap analysis.

## Ethical Disclaimer

This project is for **educational and portfolio purposes only**.

The ATS extraction capability in `ats_mcp_server.py` is implemented exclusively to demonstrate data engineering, retrieval, and LLM orchestration techniques in a local development environment. Job postings are fetched and processed in-memory at runtime for analysis and are not persisted for commercial use, redistribution, or resale.

Users are solely responsible for operating this code in compliance with applicable laws, platform Terms of Service, and rate/access restrictions.

## Architecture Overview

### 1) Static RAG Pipeline (`Analyze Resume`)

1. `build_vector_db.py` loads `data/jobs.json` and builds a persistent FAISS index on disk.
2. `app.py` (or `analyze_cv.py`) extracts PDF text via `pypdf`.
3. Resume text is embedded and queried against FAISS (`k=1`).
4. The matched job context is analyzed by `gpt-4o` with Pydantic structured output.
5. UI renders matching skills, missing skills, and a rewritten professional summary.

### 2) Live ATS Search + Re-Ranked RAG (`Search Live ATS Jobs`)

`run_live_job_search()` in `live_jobs.py` implements a multi-stage pipeline:

1. **CV profile distillation** (`gpt-4o-mini`) — extracts a compact technical summary **and** a set of high-signal `title_keywords` (e.g., `["Scientist", "Machine Learning", "Algorithm"]`) used as discriminative filters in the next step.
2. **ATS MCP extraction** — the local `ats_mcp_server.py` MCP server is invoked via `langchain-mcp-adapters`; all 10 configured Greenhouse board URLs are queried in parallel with `asyncio.gather`. Each board is fetched via the public Greenhouse Jobs API (`boards-api.greenhouse.io`).
3. **Pool shuffle and cap** — all collected jobs are shuffled with `random.shuffle` before the `_MAX_JOBS` cap is applied, preventing any single company from dominating the candidate pool.
4. **High-signal title keyword filter** — each job title is checked (case-insensitive substring match) against the LLM-generated `title_keywords`. Jobs with no matching keyword are dropped. If the filter would eliminate everything, a safety net passes all jobs through. Keywords are deliberately chosen to be discriminative nouns and compound phrases (e.g., `"Scientist"`, `"Machine Learning"`); generic words like `"Data"`, `"Engineer"`, and `"Manager"` are explicitly forbidden by the prompt to prevent false positives.
5. **Language normalization** — each description is translated to English when Hebrew or mixed text is detected (`gpt-4o-mini`), run concurrently with `asyncio.gather`.
6. **Three-tier suitability classification** (`gpt-4o-mini`) — postings are labelled `suitable`, `borderline`, or `reject` based on required day-to-day skills (not job title). Analyst/BI/junior-heavy postings are filtered out. A tiered safety net progressively relaxes the threshold if fewer than 3 suitable jobs pass.
7. **In-memory FAISS index** — qualified job descriptions are embedded with `OpenAIEmbeddings` and stored in a FAISS index using `DistanceStrategy.COSINE` (required for un-normalized OpenAI embeddings; inner-product distance on unit vectors equals cosine similarity).
8. **HyDE retrieval** — instead of querying with the raw CV profile, `gpt-4o-mini` generates a hypothetical ideal job description (Hypothetical Document Embedding) that matches the candidate. This query is issued in job-description embedding space, dramatically improving retrieval precision for asymmetric corpora. Top 20 candidates are retrieved.
9. **Domain gate + fit scoring** (`gpt-4o-mini`) — each candidate receives a structured `_JobScore` with three fields evaluated in order:
   - `is_same_domain` (bool) — if `False` (e.g., a Sales or HR role for a Data Scientist), the score is **hard-capped at 1** with no exceptions. This prevents off-domain roles from ever surfacing as results.
   - `score` (int 1–10) — combined Rigor × Candidate Fit score, only applied when `is_same_domain` is `True`.
   - `rationale` (str) — one-sentence explanation of the verdict.
10. **Deep fit/gap analysis** (`gpt-4o`) — only the top 3 scored jobs receive a full `matching_skills` / `missing_skills` / `summary` analysis. This keeps the expensive model usage tightly gated.

### Model allocation

| Model | Used for |
|---|---|
| `gpt-4o-mini` | CV profile + keyword extraction, translation, suitability classification, HyDE query generation, domain gate + scoring |
| `gpt-4o` | Deep gap analysis on top 3 results only |

### MCP server (`ats_mcp_server.py`)

A locally-run FastMCP server exposing a single tool: `extract_jobs_from_ats(url)`.

- Detects the ATS provider from the URL (Greenhouse or Comeet).
- For **Greenhouse**: calls the public Jobs API (`boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true`), strips HTML from descriptions.
- For **Comeet**: resolves `company_uid` and `token` from URL path or page HTML, then calls the Comeet careers API. Currently not used (Comeet endpoints returning HTTP 400 as of 2026-05-07).
- Returns a JSON string (`[{title, location, description, url}, ...]`).

`live_jobs.py` connects to this server via stdio transport using `MultiServerMCPClient`.

## Repository Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — static analysis and live ATS search |
| `live_jobs.py` | Full live-search pipeline: ATS extraction, filtering, retrieval, scoring, analysis |
| `ats_mcp_server.py` | Local MCP server wrapping Greenhouse/Comeet public APIs |
| `build_vector_db.py` | Offline FAISS index build from `data/jobs.json` |
| `analyze_cv.py` | CLI version of the static RAG analysis |
| `test_connection.py` | Verifies `OPENAI_API_KEY` is valid and reachable |

## Technology Stack

- Python 3.12
- Streamlit
- LangChain (`langchain-openai`, `langchain-community`, `langchain-mcp-adapters`)
- OpenAI: `gpt-4o-mini` (triage/scoring/translation), `gpt-4o` (deep analysis), `text-embedding-ada-002` (FAISS)
- FAISS (`faiss-cpu`) with `DistanceStrategy.COSINE`
- FastMCP (`mcp`)
- `httpx` (ATS API calls in `ats_mcp_server.py`)
- `pypdf`
- `python-dotenv`
- `uv` for dependency and environment management

## Setup

Prerequisites: Python 3.12+, [`uv`](https://docs.astral.sh/uv/getting-started/installation/), `OPENAI_API_KEY`.

```bash
uv sync
```

Create a `.env` file at the project root:

```
OPENAI_API_KEY=sk-...
```

## Running

Build the static FAISS index (required for the static analysis path only):

```bash
uv run python build_vector_db.py
```

Launch the Streamlit app:

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

- The live ATS path requires no prebuilt index — it builds an ephemeral in-memory FAISS index at search time.
- The static path requires a prebuilt `faiss_index/` directory (run `build_vector_db.py` first).
- `data/` and `faiss_index/` are intentionally excluded from version control.
- ATS board availability depends on external API uptime. Boards returning HTTP 4xx are silently skipped; the pipeline raises only if **all** boards fail.
- The `_MAX_JOBS` cap (default 60) is applied after shuffling so no single board can fill the entire pool.
- To add new Greenhouse boards, append to `_SEED_ATS_URLS` in `live_jobs.py`. The Greenhouse board token is the path segment after `boards.greenhouse.io/`.
