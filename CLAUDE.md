# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
uv sync                                        # Install all dependencies
uv run python build_vector_db.py               # Build FAISS index from data/jobs.json (run once)
uv run python analyze_cv.py                    # Analyze default resume (data/my_resume/havi_resume.pdf)
uv run python analyze_cv.py path/to/resume.pdf # Analyze a specific resume
uv run streamlit run app.py                    # Launch the Streamlit web UI
uv run python test_connection.py               # Verify OpenAI API key is valid and reachable
```

## Architecture

The project is a two-stage pipeline with a web UI on top:

**Stage 1 — `build_vector_db.py`**: Reads job descriptions from `data/jobs.json`, embeds them using `OpenAIEmbeddings`, and saves a FAISS index to `faiss_index/`. Re-run whenever `data/jobs.json` changes.

**Stage 2 — `analyze_cv.py`**: Loads the FAISS index, extracts text from a resume PDF via `pypdf`, finds the best-matching job via similarity search, then calls `gpt-4o` with `with_structured_output` (Pydantic model) to return matching skills, skill gaps, and a rewritten summary. Resume path defaults to `data/my_resume/havi_resume.pdf` but can be overridden via CLI argument.

**`app.py`**: Streamlit web UI wrapping the same logic as `analyze_cv.py`. Provides a file uploader for the resume PDF and displays results in a two-column layout (matching skills / skill gaps) plus a rewritten summary card.

**`test_connection.py`**: Standalone utility that validates the `OPENAI_API_KEY` format and attempts to instantiate `OpenAIEmbeddings`.

## Adding Job Descriptions

Edit `data/jobs.json` — each entry needs a `"title"` and `"description"` field — then re-run `build_vector_db.py` to rebuild the index.

## Environment

A `.env` file at the project root must contain:
```
OPENAI_API_KEY=sk-...
```

`analyze_cv.py` and `app.py` both require the `faiss_index/` directory to exist before running.
