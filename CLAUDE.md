# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Disclaimer

This project is for **educational and portfolio purposes only**.

The ATS extraction implementation in `ats_mcp_server.py` exists solely to demonstrate data engineering and LLM integration patterns in local development. Fetched job postings are processed in-memory at runtime for analysis and are not intended for commercial redistribution, resale, or persistent dataset creation.

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
- `analyze_cv.py` (and the corresponding `app.py` branch) extracts PDF text, retrieves the best match from FAISS (`k=1`), and runs `gpt-4o` with structured output for:
  - `matching_skills`
  - `missing_skills`
  - `rewritten_summary`

### 2) Live ATS search + re-ranked RAG path (`Search Live ATS Jobs`)

Implemented in `live_jobs.py` through `run_live_job_search()`:

1. **CV profile distillation** (`gpt-4o-mini`) — produces a `_CvProfile` with two fields:
   - `summary`: compact technical profile used as context in all downstream LLM calls.
   - `title_keywords`: 3–5 high-signal, discriminative keywords (e.g., `["Scientist", "Machine Learning", "Algorithm"]`) derived from the candidate's actual job title and core expertise. Used in step 3b.

2. **ATS MCP extraction** — `_discover_jobs_via_ats_mcp()` opens a stdio session to `ats_mcp_server.py` via `MultiServerMCPClient`. All seed Greenhouse board URLs in `_SEED_ATS_URLS` are queried in parallel with `asyncio.gather`. The MCP tool `extract_jobs_from_ats(url)` calls the public Greenhouse Jobs API.

3. **Pool shuffle and cap** — all collected jobs are passed through `random.shuffle()` before being sliced to `_MAX_JOBS` (currently 60). This prevents a high-volume board from dominating the pool.

3b. **High-signal title keyword filter** — `_filter_by_title_keywords()` retains only jobs whose title contains at least one of the `title_keywords` (case-insensitive substring match). Safety net: if zero jobs match, all jobs are passed through to avoid an empty pipeline. **Keyword generation rules enforced by prompt:**
   - The first keyword MUST be the core job-title noun (e.g., `"Scientist"`, `"Researcher"`).
   - Industry domains (`"FinTech"`, `"Cyber"`) and stack tools (`"NLP"`, `"Deep Learning"`) are forbidden — they appear in descriptions, not titles.
   - Generic words (`"Data"`, `"Engineer"`, `"Manager"`, `"Analyst"`) are forbidden — they cause massive false positives (e.g., `"Data"` matches `"Data Entry"`).

4. **Language normalization** (`gpt-4o-mini`) — descriptions in Hebrew or mixed Hebrew/English are translated to English concurrently via `asyncio.gather`. Pure-English descriptions are returned unchanged.

5. **Three-tier suitability classification** (`gpt-4o-mini`) — each job description is classified as `suitable`, `borderline`, or `reject` based on required day-to-day skills (not job title). BI/reporting/junior/data-engineering-only roles are hard-rejected. Tiered safety net: if fewer than 3 jobs reach `suitable`, `borderline` jobs are added; if still fewer than 3, hard-rejects are included too.

6. **In-memory FAISS index** — qualified job descriptions are embedded with `OpenAIEmbeddings` and indexed using `DistanceStrategy.COSINE`. Cosine distance is required because OpenAI embeddings are not unit-normalized; using the default inner-product distance would produce incorrect similarity rankings.

7. **HyDE retrieval** — `_build_hyde_query()` (`gpt-4o-mini`) generates a synthetic "ideal job description" for the candidate (Hypothetical Document Embedding). This is used as the FAISS query instead of the raw CV profile, because querying in job-description embedding space is far more precise than querying with a CV-style text for asymmetric corpora. Top `k=20` candidates are retrieved.

8. **Domain gate + fit scoring** (`gpt-4o-mini`) — `_score_job()` uses a `_JobScore` Pydantic model with three fields evaluated in strict order:
   - `is_same_domain` (bool): set **before** `score`. If `False` (e.g., a Sales, HR, or Marketing role for a Data Scientist), `score` is **hard-capped at 1** unconditionally. No tenuous connections (e.g., "this sales role uses CRM data") are permitted.
   - `score` (int 1–10): combined Rigor × Candidate Fit, evaluated only when `is_same_domain` is `True`. Axis 1 measures ML/DS depth of the role; Axis 2 measures stack/domain match with the candidate.
   - `rationale` (str): one-sentence explanation of the verdict.
   Jobs are sorted by score descending; top 3 above a minimum threshold are selected.

9–11. **Deep gap analysis** (`gpt-4o`) — `_analyze_match()` is called only on the top 3 scored jobs, returning `matching_skills`, `missing_skills`, and a `summary`. This keeps expensive model usage tightly gated.

### Model allocation strategy

| Model | Responsibility |
|---|---|
| `gpt-4o-mini` | CV profile + keyword extraction, translation, suitability classification, HyDE query generation, domain gate + scoring |
| `gpt-4o` | Deep gap analysis on top 3 results only |

### MCP server (`ats_mcp_server.py`)

A locally-run FastMCP server that exposes one tool: `extract_jobs_from_ats(url: str) -> str`.

- Detects ATS type from URL host/path (`greenhouse` or `comeet`).
- **Greenhouse**: calls `boards-api.greenhouse.io/v1/boards/{token}/jobs?content=true`, strips HTML from `content` field.
- **Comeet**: resolves `company_uid` and `token` from URL path or embedded page JavaScript, then calls `comeet.co/careers-api/2.0/company/{uid}/positions`. Currently not used — Comeet endpoints returning HTTP 400 as of 2026-05-07; all Comeet entries have been removed from `_SEED_ATS_URLS`.
- Returns a JSON string: `[{"title": ..., "location": ..., "description": ..., "url": ...}, ...]`.

`live_jobs.py` connects via stdio transport using `MultiServerMCPClient` from `langchain-mcp-adapters`.

## Key implementation details

- **`_SEED_ATS_URLS`** in `live_jobs.py` — Greenhouse boards only. Add new entries as `"https://boards.greenhouse.io/{board_token}"`. The board token is the company-specific path segment. Entries returning HTTP 4xx are silently skipped.
- **`_MAX_JOBS`** — currently `60`. Applied after `random.shuffle()`, not before.
- **`_filter_by_title_keywords`** — both the job title and the keywords are lowercased before comparison (`kw.lower() in title.lower()`). The filter is a no-op if `title_keywords` is empty.
- **FAISS distance strategy** — always `DistanceStrategy.COSINE` in the live path. Do not change to `EUCLIDEAN_DISTANCE` or the default without understanding the implications for un-normalized embeddings.
- **Structured outputs** — all LLM calls that require parsed fields use `.with_structured_output(PydanticModel)`. Never switch these to unstructured `.invoke()` calls without adding explicit parsing.
- **Async boundary** — `_discover_jobs_via_ats_mcp()` is an async coroutine. It is run from synchronous Streamlit context via `_run_async()`, which spawns a dedicated thread with its own event loop. Do not call it directly from a synchronous context.

## Adding Job Descriptions (Static Path)

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

- `analyze_cv.py` and the static `app.py` path require `faiss_index/` to exist (run `build_vector_db.py` first).
- The live ATS path does not require the static index, but requires network access and a valid OpenAI API key.
- `data/` and `faiss_index/` are excluded from version control.
