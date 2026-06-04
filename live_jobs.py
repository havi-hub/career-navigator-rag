import asyncio
import concurrent.futures
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pydantic import BaseModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

load_dotenv()

_MINI = "gpt-4o-mini"   # profiling + triage: cheap
_FULL = "gpt-4o"         # deep gap analysis on top 3 only: expensive

_MAX_JOBS = 60  # broader pool improves coverage; translation/classification cap is upstream

# Seed ATS boards — Greenhouse only (Comeet removed: returning HTTP 400).
# Add the exact careers-board URL for any company you want to target.
_SEED_ATS_URLS: list[str] = [
    # Greenhouse — Israel-based AI/ML/DS roles
    # Last manually verified: 2026-05-07.
    "https://boards.greenhouse.io/taboola",
    "https://boards.greenhouse.io/appsflyer",
    "https://boards.greenhouse.io/similarweb",
    "https://boards.greenhouse.io/forter",
    "https://boards.greenhouse.io/payoneer",
    "https://boards.greenhouse.io/lightricks",
    "https://boards.greenhouse.io/unframe",
    "https://boards.greenhouse.io/nift",
    "https://boards.greenhouse.io/fireblocks",
    "https://boards.greenhouse.io/riskified",
]


def _infer_company_name(job_url: str, source_url: str) -> str:
    """Infer a simple company label from ATS URLs."""
    for candidate in (job_url, source_url):
        parsed = urlparse(candidate or "")
        host = (parsed.hostname or "").lower()
        parts = [p for p in (parsed.path or "").split("/") if p]

        if "greenhouse.io" in host and parts:
            return parts[0]
        if "comeet.co" in host and len(parts) >= 2 and parts[0] == "jobs":
            return parts[1]

    return "Unknown"


def _translate_description(text: str) -> str:
    """
    Detect language and translate to English if needed. No-op for English-only text.
    Called via asyncio.to_thread so it doesn't block the event loop.
    """
    llm = ChatOpenAI(model=_MINI)
    prompt = f"""Detect the language of the text below.
- If it is already entirely in English, return it EXACTLY as provided with no changes whatsoever.
- If it contains any Hebrew (or is mixed Hebrew/English), translate the ENTIRE text into professional technical English. Preserve all technical terms, tool names, and proper nouns exactly as written.

Text:
{text}
"""
    return llm.invoke(prompt).content.strip()


async def _translate_to_english(job: dict) -> dict:
    """Run description translation in a thread so the sync LLM call doesn't block asyncio.gather."""
    translated = await asyncio.to_thread(_translate_description, job["description"])
    return {**job, "description": translated}


async def _translate_job_title(job: dict) -> dict:
    """Translate only the title field. Titles are short so this is cheap."""
    translated = await asyncio.to_thread(_translate_description, job.get("title") or "")
    return {**job, "title": translated}

def _pick_ats_extract_tool(tools: list[Any]) -> Any:
    """Select the MCP tool wrapper named `extract_jobs_from_ats`."""
    for t in tools:
        if getattr(t, "name", None) == "extract_jobs_from_ats":
            return t
    if tools:
        return tools[0]
    raise RuntimeError("No MCP tools were loaded from ats_mcp_server.py")


async def _call_extract_jobs_from_ats(tool: Any, url: str) -> list[dict[str, Any]]:
    """Invoke the MCP tool wrapper and parse its JSON-string output."""
    # LangChain tool wrappers generally want a dict input: {"url": "..."}.
    raw: Any
    if hasattr(tool, "ainvoke"):
        raw = await tool.ainvoke({"url": url})
    elif hasattr(tool, "arun"):
        raw = await tool.arun(url)
    else:
        raw = tool.invoke({"url": url})

    # The MCP server returns a JSON string, but LangChain/MCP adapters can wrap
    # this in several response envelopes depending on version/runtime.
    if isinstance(raw, str):
        return json.loads(raw)

    if isinstance(raw, list):
        # MCP adapter may return content blocks:
        #   [{"type": "text", "text": "[{...jobs...}]"}]
        if raw and all(isinstance(x, dict) and "type" in x for x in raw):
            for block in raw:
                text = block.get("text")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            return parsed
                    except Exception:
                        continue
        return raw

    if isinstance(raw, dict):
        # Sometimes adapters return {"content": [...]} where text payload is in
        # content blocks (or structured content under an artifact-like key).
        content = raw.get("content")
        if isinstance(content, str):
            return json.loads(content)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        return json.loads(text)
        return raw.get("structured_content") or raw.get("structuredContent") or []

    # ToolMessage-style objects
    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return json.loads(content)
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    return json.loads(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    return json.loads(text)

    # Last resort: attempt JSON parsing from string representation.
    return json.loads(str(raw))


async def _discover_jobs_via_ats_mcp(title_keywords: list[str]) -> list[dict[str, Any]]:
    """
    Job discovery via ATS MCP server:
      1) query all seed Greenhouse boards in parallel
      2) deduplicate and shuffle
      3) translate titles to English (so keyword filter works on non-English titles)
      4) keyword filter, then cap at _MAX_JOBS
      5) drop short descriptions
      6) translate descriptions to English
    """

    # 1) Query all seed boards — no LLM selection needed, always use full list.
    selected = list(range(len(_SEED_ATS_URLS)))

    # 2) Connect to local MCP server and call the tool for each chosen seed URL.
    server_path = Path(__file__).with_name("ats_mcp_server.py")
    if not server_path.exists():
        raise FileNotFoundError(f"Expected MCP server at {server_path}")

    client = MultiServerMCPClient(
        {
            "ats": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(server_path)],
            }
        }
    )

    async with client.session("ats") as session:
        tools = await load_mcp_tools(session)
        tool = _pick_ats_extract_tool(tools)

        selected_urls = [_SEED_ATS_URLS[i] for i in selected]
        tasks = [_call_extract_jobs_from_ats(tool, u) for u in selected_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    all_jobs: list[dict[str, Any]] = []
    seen: set[str] = set()

    failures: list[str] = []
    for idx, r in enumerate(results):
        if isinstance(r, Exception):
            failures.append(f"{selected_urls[idx]} -> {type(r).__name__}: {r}")
            continue
        source_url = selected_urls[idx]
        for job in r:
            if not isinstance(job, dict):
                continue
            job.setdefault("company", _infer_company_name(str(job.get("url") or ""), source_url))
            job_url = str(job.get("url") or "")
            key = job_url or json.dumps(job, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            all_jobs.append(job)

    # Shuffle so no single board dominates the pool.
    random.shuffle(all_jobs)

    if not all_jobs and failures:
        raise RuntimeError(
            "MCP ATS extraction failed for all selected boards:\n"
            + "\n".join(failures[:5])
        )

    # 3) Translate titles before filtering so non-English titles match keywords correctly.
    title_tasks = [_translate_job_title(j) for j in all_jobs]
    title_results = await asyncio.gather(*title_tasks, return_exceptions=True)
    all_jobs = [j for j in title_results if not isinstance(j, Exception)]

    # 4) Keyword filter on translated titles, then cap — relevant jobs are never cut first.
    filtered = _filter_by_title_keywords(all_jobs, title_keywords)
    if not filtered:
        filtered = all_jobs  # safety net: if nothing matched, pass everything through
    filtered = filtered[:_MAX_JOBS]

    # 5) Drop short descriptions.
    valid = [j for j in filtered if len(str(j.get("description") or "")) > 200]
    if not valid:
        return []

    # 6) Translate descriptions to English.
    desc_tasks = [_translate_to_english(j) for j in valid]
    desc_results = await asyncio.gather(*desc_tasks, return_exceptions=True)
    return [j for j in desc_results if not isinstance(j, Exception)]


def _run_async(coro):
    """Run an async coroutine from a synchronous context (Streamlit runs its own loop)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── LLM helpers ────────────────────────────────────────────────────────────────

class _CvProfile(BaseModel):
    summary: str              # concise technical profile for FAISS queries and LLM prompts
    title_keywords: list[str] # 3-5 high-signal discriminative keywords for title filtering


def _build_cv_profile(resume_text: str) -> _CvProfile:
    """Distil a focused technical profile and high-signal title keywords. Uses gpt-4o-mini (cheap)."""
    llm = ChatOpenAI(model=_MINI).with_structured_output(_CvProfile)
    prompt = f"""Extract a concise technical profile from this resume strictly for job-matching purposes.

Return two fields:

summary — max 150 words, include ONLY:
  1. Seniority level and years of experience
  2. Core technical skills: ML/DL frameworks, languages, cloud tools
  3. Domain expertise: NLP, computer vision, time-series, etc.
  4. Nature of work: research, production pipelines, end-to-end, etc.
  Omit personal info, education details, and soft skills.

title_keywords — 3-5 HIGH-SIGNAL, DISCRIMINATIVE keywords used as case-insensitive substrings
  to filter live ATS job titles. You are matching against titles like "Senior Data Scientist"
  or "Machine Learning Engineer" — NOT against job descriptions.

  RULE 1 — Core noun first: Your FIRST keyword MUST be the core job-title noun from the
  candidate's resume header (e.g., "Scientist", "Researcher", "Architect", "Algorithm").
  This single word is the most reliable signal because it appears in every relevant title.

  RULE 2 — Titles only, no descriptions: DO NOT output industry domains ("FinTech", "Cyber",
  "Healthcare") or technology stack terms ("NLP", "Deep Learning", "Python", "TensorFlow").
  Those words appear in job descriptions, not titles. Using them here produces zero matches.

  RULE 3 — No generic words: DO NOT output "Data", "Engineer", "Developer", "Manager",
  "Lead", "Senior", "Junior", or "Analyst". "Data" alone matches "Data Entry" and
  "Data Quality" — roles that are completely irrelevant for an ML candidate.

  CORRECT example — Senior Data Scientist with ML/NLP background:
    ✅ ["Scientist", "Machine Learning", "Algorithm", "AI"]
    ❌ ["FinTech", "NLP", "Deep Learning", "Research"]  ← industry domain + stack terms, not titles

  More examples by role:
    NLP/LLM specialist  → ["Scientist", "Language Model", "Conversational AI", "Algorithm"]
    CV engineer         → ["Scientist", "Computer Vision", "Perception", "Machine Learning"]
    MLOps practitioner  → ["MLOps", "Machine Learning", "AI Platform", "Scientist"]

Resume:
{resume_text[:4000]}
"""
    return llm.invoke(prompt)


def _build_hyde_query(cv_profile: str) -> str:
    """
    HyDE (Hypothetical Document Embeddings): generate a synthetic job description
    that would be a perfect fit for this candidate, then embed IT as the FAISS query.
    Querying in job-description space instead of CV-profile space dramatically
    improves retrieval precision for asymmetric corpora.
    """
    llm = ChatOpenAI(model=_MINI)
    prompt = f"""Write a realistic, detailed job posting for a senior ML / Data Science role
that would be a perfect fit for a candidate with the profile below.

Include:
- A "Responsibilities" section describing day-to-day work
- A "Requirements" section listing must-have and nice-to-have skills
- Use the candidate's actual stack and domain so the posting closely mirrors real roles they'd excel in.

Max 300 words. Do NOT mention the candidate by name.

Candidate profile:
{cv_profile}
"""
    return llm.invoke(prompt).content.strip()


def _filter_by_title_keywords(jobs: list[dict], title_keywords: list[str]) -> list[dict]:
    """Keep only jobs whose title contains at least one high-signal keyword (case-insensitive)."""
    if not title_keywords:
        return jobs
    keywords_lower = [kw.lower() for kw in title_keywords]
    return [
        j for j in jobs
        if any(kw in (j.get("title") or "").lower() for kw in keywords_lower)
    ]


class _JobClassification(BaseModel):
    role_type: str
    suitability: Literal["suitable", "borderline", "reject"]
    reason: str

def _classify_job(job_description: str, cv_profile: str) -> _JobClassification:
    """Gate-keep scraped jobs on required SKILLS, not job title. Uses gpt-4o-mini (cheap)."""
    llm = ChatOpenAI(model=_MINI).with_structured_output(_JobClassification)
    prompt = f"""You are a strict technical recruiter screening job postings for a senior machine-learning practitioner.

CRITICAL RULE: Ignore the job title completely. Judge solely on the day-to-day skills and tools
the posting actually requires. A posting titled "Data Scientist" that only asks for SQL and
dashboards is NOT suitable. A posting titled "Data Analyst" that requires building and deploying
predictive models IS suitable.

HARD REJECT (is_suitable = False) when the primary day-to-day work is ANY of:
- BI / reporting tools: Tableau, Power BI, Looker, Qlik, Excel, Google Data Studio
- Writing SQL queries, building data pipelines, or ETL without ML modelling
- Describing, summarising, or visualising data (descriptive analytics)
- A/B test analysis, KPI tracking, or business dashboards
- Data Engineering roles focused on Spark/Kafka/Airflow/dbt/warehousing without owning model development
- Backend / software engineering roles (APIs, microservices, distributed systems) without core ML modelling responsibilities
- No explicit mention of model building, training, or evaluation
- Requirements capped at 0-2 years of experience

STRONGLY ACCEPT (is_suitable = True) ONLY when the job explicitly and primarily requires:
- Classical ML: regression, classification, clustering, ensembles, feature engineering at scale
- Deep Learning: neural networks, transformers, PyTorch, TensorFlow, JAX
- Generative AI / LLMs: fine-tuning, RAG, prompt engineering in production, RLHF
- NLP: text modelling, embeddings, NER, summarisation, language models
- Computer vision: CNNs, object detection, image segmentation
- MLOps / model production: serving, monitoring, retraining pipelines, CI/CD for models
- Advanced probabilistic or statistical modelling (Bayesian inference, causal models, etc.)
- Predictive modelling ownership: defining targets, building features, offline/online evaluation, and iteration

SUITABILITY LEVELS — assign exactly one:
  "suitable"   — role explicitly and primarily requires work from the STRONGLY ACCEPT list above.
  "borderline" — role has real ML/modelling components but they are mixed with significant SQL/reporting/infra work, OR the posting is vague but not clearly a hard reject.
  "reject"     — role is primarily BI, reporting, data engineering, backend, or junior-level without meaningful modelling ownership.

PRIORITIZATION RULE:
- Favor classic ML/Data Science roles over generic "AI" wrapper roles.
- If "AI" appears but work is mostly integrations, product glue, prompt ops, or backend software, use "borderline" or "reject".

When in doubt between "suitable" and "borderline", prefer "borderline".
Only use "reject" for clear hard-rejects (BI tools, SQL-only, no modelling at all).

Candidate profile (for seniority reference only — do not let it override the skill rules above):
{cv_profile}

Job description to evaluate:
{job_description[:2500]}

Return:
- role_type: honest short label based on actual required skills, e.g. "Senior DS", "ML Engineer",
  "NLP Researcher", "MLOps Engineer", "Data Analyst", "BI Analyst", "Junior DS"
- suitability: "suitable" | "borderline" | "reject"
- reason: one sentence citing the specific skills (or lack thereof) that drove the decision
"""
    return llm.invoke(prompt)


class _JobScore(BaseModel):
    is_same_domain: bool  # MUST be set before score — drives the domain-rejection gate
    score: int            # 1-10 combined Rigor × Fit score (forced to 1 if is_same_domain=False)
    rationale: str        # one sentence explaining the score


def _score_job(job_description: str, cv_profile: str) -> _JobScore:
    """
    Assign a combined Fit Score (1-10) on two axes:
      - Scientific Rigor: how ML/DS-heavy is the day-to-day work?
      - Candidate Match: how well do the required skills align with THIS candidate's profile?
    Domain gate fires first: off-domain roles are hard-capped at 1. Uses gpt-4o-mini.
    """
    llm = ChatOpenAI(model=_MINI).with_structured_output(_JobScore)
    prompt = f"""You are a senior ML hiring manager scoring job postings for a specific candidate.

STEP 1 — Domain gate (evaluate this before anything else):
Determine whether the job is in the exact same professional domain as the candidate's core expertise.

Set is_same_domain = false if the job's PRIMARY day-to-day work is in a completely different
profession, such as: Sales, Account Management, Marketing, HR, Recruiting, Legal, Finance,
Customer Success, Customer Support, Advertising, or Frontend/Web Engineering — for a candidate
whose expertise is Data Science, Machine Learning, or AI Research.

If is_same_domain = false, you MUST set score = 1 with no exceptions. Do not search for
tenuous connections (e.g., "the sales role uses CRM data"). A fundamentally different profession
is a hard reject regardless of any other factor. Set rationale to one sentence explaining the
domain mismatch and stop.

STEP 2 — Score (only if is_same_domain = true):
Assign a combined Fit Score from 1 to 10 reflecting TWO equally weighted axes:

AXIS 1 — Scientific Rigor (does this role require real ML/DS work?):
  High (8-10): model training/research, GenAI/LLM/RAG/RLHF, NLP, CV, MLOps,
    advanced probabilistic modelling, full predictive-modelling lifecycle ownership.
  Medium (5-7): ML buried under SQL/reporting, data pipelines with occasional ML.
  Low (1-4): BI tools, SQL/KPI dashboards, descriptive analytics, data engineering
    without model development, backend/software with peripheral AI.

AXIS 2 — Candidate Match (does this job fit THIS candidate's specific skills?):
  High (8-10): role's required stack and domain closely mirrors the candidate's experience.
  Medium (5-7): meaningful overlap but notable gaps in stack, domain, or seniority.
  Low (1-4): large mismatch in required skills, domain, or seniority level.

RULES for Step 2:
- Ignore job title. Judge on actual required day-to-day work.
- A posting requiring skills the candidate lacks must not score above 6.
- A well-matched role where the candidate ticks 80%+ of requirements should score 8-10.
- Combine both axes: a rigorous ML job with poor candidate fit ≤ a medium-rigor job with perfect fit.

Candidate profile:
{cv_profile}

Job description:
{job_description[:3000]}

Return:
- is_same_domain: true if the job's core profession matches the candidate's domain, else false
- score: integer 1-10 (must be 1 when is_same_domain is false)
- rationale: one sentence explaining the domain verdict and/or how rigor × fit produced this score
"""
    return llm.invoke(prompt)


class LiveJobAnalysis(BaseModel):
    matching_skills: list[str]
    missing_skills: list[str]
    summary: str


def _analyze_match(resume_text: str, job_description: str) -> LiveJobAnalysis:
    """Deep gap analysis — only called on top 3 matches. Uses gpt-4o (expensive)."""
    llm = ChatOpenAI(model=_FULL).with_structured_output(LiveJobAnalysis)
    prompt = f"""You are an expert career coach and resume analyst.

Resume:
---
{resume_text[:3000]}
---

Job Description:
---
{job_description[:3000]}
---

Return exactly:
- matching_skills: top 5 skills/technologies the resume shares with this job
- missing_skills: top 5 skills or qualifications the resume lacks for this role
- summary: 2-3 sentences explaining why this is a strong, partial, or poor fit
"""
    return llm.invoke(prompt)


# ── Main pipeline ──────────────────────────────────────────────────────────────

def validate_results(results: list[dict]) -> list[str]:
    """
    Structural sanity checks on the output of run_live_job_search().
    Returns a list of violation strings; empty list means all checks passed.
    Does not re-run the pipeline — checks only the shape and invariants of the output.
    """
    violations: list[str] = []

    if not results:
        violations.append("results list is empty — pipeline returned no jobs")
        return violations  # remaining checks require at least one result

    for i, job in enumerate(results):
        label = f"result[{i}] ({job.get('title', '?')})"

        score = job.get("score")
        if not isinstance(score, int) or not (1 <= score <= 10):
            violations.append(f"{label}: score={score!r} is not an integer in [1, 10]")

        if not isinstance(job.get("matching_skills"), list) or not job["matching_skills"]:
            violations.append(f"{label}: matching_skills is empty or missing")

        if not isinstance(job.get("missing_skills"), list) or not job["missing_skills"]:
            violations.append(f"{label}: missing_skills is empty or missing")

        if not job.get("summary", "").strip():
            violations.append(f"{label}: summary is blank")

        if job.get("is_same_domain") is False and score != 1:
            violations.append(
                f"{label}: is_same_domain=False but score={score} (must be 1)"
            )

    return violations


def run_live_job_search(resume_text: str, progress_callback=None) -> tuple[list[dict], str]:
    """
    Optimised pipeline:
      1.  GPT-4o Mini — distil CV profile (cheap triage)
      2.  MCP         — query ATS seed boards and extract open jobs (asyncio.gather)
      3.  GPT-4o Mini — translate Hebrew/mixed descriptions to English CONCURRENTLY
      4.  GPT-4o Mini — three-tier classify (suitable / borderline / reject); tiered safety net
      5.  OpenAI      — build in-memory FAISS index with cosine similarity on qualified jobs
      6.  GPT-4o Mini — HyDE: generate hypothetical ideal job description as retrieval query
      7.  FAISS       — retrieve top 20 candidates by cosine similarity against HyDE query
      8.  GPT-4o Mini — candidate-aware Fit Score (rigor × match, 1-10); keep top 3; score=0 on failure
      9-11. GPT-4o   — deep gap analysis on top 3 scored jobs only (expensive, targeted)
    """
    TOTAL = 11
    step = 0

    def _progress(msg: str):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, TOTAL, msg)

    # 1 — distil CV profile with gpt-4o-mini
    _progress("Extracting your technical profile with GPT-4o Mini...")
    profile = _build_cv_profile(resume_text)
    cv_profile = profile.summary   # string alias used throughout downstream helpers
    print(f"[DEBUG] Title keywords: {profile.title_keywords}")

    # 2 — query ATS MCP server: fetch all boards, translate titles, filter by keyword, cap, translate descriptions
    _progress("Querying ATS MCP server to extract live jobs...")
    raw_jobs = _run_async(_discover_jobs_via_ats_mcp(profile.title_keywords))

    # 3 — extraction + filtering + translation complete; report results
    _progress(f"Extracted and translated {len(raw_jobs)} job postings. Processing descriptions...")

    if not raw_jobs:
        raise RuntimeError(
            "ATS MCP found no job descriptions. "
            "Try adding seed ATS URLs or check that the boards have published roles."
        )

    # 4 — three-tier classify with gpt-4o-mini; tiered safety net prevents over-filtering
    _progress(f"Classifying {len(raw_jobs)} jobs with GPT-4o Mini (filtering analyst/junior roles)...")

    classified: list[tuple[dict, str]] = []   # (job, suitability level)
    for job in raw_jobs:
        try:
            clf = _classify_job(job["description"], cv_profile)
            if clf.suitability == "suitable":
                job["role_type"] = clf.role_type
            classified.append((job, clf.suitability))
        except Exception:
            classified.append((job, "suitable"))  # keep on classifier error

    # Tier 1: only strongly suitable jobs
    qualified = [j for j, s in classified if s == "suitable"]
    # Tier 2: add borderline if fewer than 3 passed
    if len(qualified) < 3:
        qualified = [j for j, s in classified if s in ("suitable", "borderline")]
    # Tier 3: exclude only hard rejects — never silently drop all filtering
    if len(qualified) < 3:
        qualified = [j for j, s in classified if s != "reject"]
    # Last resort: use everything (original safety net, only fires if all jobs were hard-rejected)
    if len(qualified) < 3:
        qualified = raw_jobs

    # 5 — build FAISS index with cosine similarity on qualified jobs only
    _progress(f"Building cosine-similarity FAISS index from {len(qualified)} qualified jobs...")
    embeddings = OpenAIEmbeddings()
    docs = [
        Document(
            page_content=j["description"],
            metadata={
                "title": j["title"],
                "url": j["url"],
                "company": j.get("company", "Unknown"),
            },
        )
        for j in qualified
    ]
    vector_store = FAISS.from_documents(
        docs, embedding=embeddings, distance_strategy=DistanceStrategy.COSINE
    )

    # 6 — HyDE: generate a hypothetical ideal job description and use it as the search query
    _progress("Generating optimized semantic search query with HyDE technique...")
    hyde_query = _build_hyde_query(cv_profile)

    # 7 — retrieve broad candidate pool from FAISS (k=20 so scorer has real breadth)
    _progress("Retrieving top 20 candidates by cosine similarity...")
    candidates = vector_store.similarity_search(hyde_query, k=min(20, len(qualified)))
    print("\n[DEBUG] FAISS retrieved candidates (before scoring):")
    for i, match in enumerate(candidates, start=1):
        print(
            f"[DEBUG] #{i} | title={match.metadata.get('title', 'Unknown')} | "
            f"company={match.metadata.get('company', 'Unknown')}"
        )

    # 8 — candidate-aware Fit Score (rigor × candidate match); failures score 0 to sink to bottom
    _progress(f"Scoring {len(candidates)} candidates for rigor + candidate fit with GPT-4o Mini...")
    scored: list[tuple[int, str, bool, object]] = []   # (score, rationale, is_same_domain, match)
    for match in candidates:
        try:
            s = _score_job(match.page_content, cv_profile)
            scored.append((s.score, s.rationale, s.is_same_domain, match))
            print(
                f"[DEBUG] SCORED | title={match.metadata.get('title', 'Unknown')} | "
                f"company={match.metadata.get('company', 'Unknown')} | "
                f"same_domain={s.is_same_domain} | score={s.score}"
            )
        except Exception:
            scored.append((0, "Scoring failed — excluded from ranking.", True, match))
            print(
                f"[DEBUG] SCORED | title={match.metadata.get('title', 'Unknown')} | "
                f"company={match.metadata.get('company', 'Unknown')} | score=0 (fallback)"
            )

    scored.sort(key=lambda x: x[0], reverse=True)

    # Only show results that genuinely pass the quality bar.
    # Progressively relax the threshold rather than silently padding with bad matches.
    for min_score in (7, 6, 5, 0):
        top3 = [x for x in scored if x[0] >= min_score][:3]
        if top3:
            break

    # 9-11 — deep gap analysis with gpt-4o (expensive — top 3 only)
    results: list[dict] = []
    for i, (score, rationale, is_same_domain, match) in enumerate(top3, start=1):
        title = match.metadata.get("title", "Unknown")
        _progress(f"Deep gap analysis {i}/3 with GPT-4o: {title[:50]}...")
        analysis = _analyze_match(resume_text, match.page_content)
        results.append({
            "title": title,
            "url": match.metadata.get("url", ""),
            "score": score,
            "is_same_domain": is_same_domain,
            "score_rationale": rationale,
            "matching_skills": analysis.matching_skills,
            "missing_skills": analysis.missing_skills,
            "summary": analysis.summary,
        })

    if progress_callback:
        progress_callback(TOTAL, TOTAL, "Done!")

    violations = validate_results(results)
    if violations:
        raise RuntimeError("Output validation failed:\n" + "\n".join(f"  • {v}" for v in violations))

    return results, profile.summary
