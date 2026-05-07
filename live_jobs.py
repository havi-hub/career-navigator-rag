import asyncio
import concurrent.futures
import urllib.parse

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from playwright.async_api import async_playwright
from playwright_stealth import Stealth
from pydantic import BaseModel

load_dotenv()

_MINI = "gpt-4o-mini"   # profiling + triage: cheap
_FULL = "gpt-4o"         # deep gap analysis on top 3 only: expensive

_BLOCKED = {"image", "media", "font", "stylesheet"}
_INDEED_BASE = "https://il.indeed.com"
_MAX_JOBS = 15   # scrape enough candidates so k=20 FAISS retrieval + scoring layer has real breadth


# ── HTML cleaning ──────────────────────────────────────────────────────────────

def _clean_html(html: str) -> str:
    """Strip boilerplate and return plain text from a job page, capped at 4000 chars."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]):
        tag.decompose()

    container = (
        soup.find("div", id="jobDescriptionText")
        or soup.find("div", attrs={"class": lambda c: c and "jobsearch-jobDescriptionText" in " ".join(c)})
        or soup.find("div", attrs={"class": lambda c: c and "job-description" in " ".join(c)})
        or soup.find("main")
        or soup.find("article")
    )
    source = container if container else soup.body
    if not source:
        return ""

    lines = [ln.strip() for ln in source.get_text(separator="\n", strip=True).splitlines() if ln.strip()]
    # Deduplicate adjacent identical lines (boilerplate repeats)
    deduped: list[str] = []
    prev = None
    for line in lines:
        if line != prev:
            deduped.append(line)
            prev = line
    return "\n".join(deduped)[:4000]


# ── Playwright async scraper ───────────────────────────────────────────────────

async def _abort_if_blocked(route):
    """Block network requests for images, media, fonts, and CSS for faster page loads."""
    if route.request.resource_type in _BLOCKED:
        await route.abort()
    else:
        await route.continue_()


async def _scrape_job_page(context, url: str, title: str) -> dict:
    page = await context.new_page()
    try:
        await page.route("**/*", _abort_if_blocked)
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        description = _clean_html(await page.content())
    except Exception:
        description = ""
    finally:
        await page.close()
    return {"title": title, "url": url, "description": description}


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
    """Run translation in a thread so the sync LLM call doesn't block asyncio.gather."""
    translated = await asyncio.to_thread(_translate_description, job["description"])
    return {**job, "description": translated}


async def _scrape_indeed(query: str = "Senior Data Scientist", location: str = "Israel") -> list[dict]:
    """
    Navigate to il.indeed.com, extract job URLs from the first page, then scrape
    up to _MAX_JOBS job pages CONCURRENTLY via asyncio.gather.
    Resource blocking (images/media/fonts/CSS) is applied on every page.
    """
    search_url = (
        f"{_INDEED_BASE}/jobs"
        f"?q={urllib.parse.quote_plus(query)}"
        f"&l={urllib.parse.quote_plus(location)}"
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        # Apply stealth evasions to every page created from this context
        await Stealth().apply_stealth_async(context)

        # --- search results page ---
        search_page = await context.new_page()
        await search_page.route("**/*", _abort_if_blocked)
        try:
            await search_page.goto(search_url, wait_until="domcontentloaded", timeout=30_000)
            # Temporary manual CAPTCHA window for Indeed anti-bot checks
            await asyncio.sleep(15)
            # Wait for actual job cards/anchors after any CAPTCHA redirect/render delay
            try:
                await search_page.wait_for_selector(
                    "div.job_seen_beacon, a[data-jk], td.resultContent",
                    timeout=15_000,
                )
            except Exception:
                # Non-fatal: we'll still parse the DOM and rely on fallbacks below
                pass
        except Exception:
            await browser.close()
            return []

        html = await search_page.content()

        # Extract job links with BeautifulSoup (no JS eval needed)
        soup = BeautifulSoup(html, "html.parser")
        entries: list[dict] = []
        seen: set[str] = set()

        # Primary + fallback selectors for evolving Indeed markup
        for a in soup.select(
            "h2.jobTitle > a, h2.jobTitle a, a.jcs-JobTitle, a[data-jk], a[id^='job_']"
        ):
            href = a.get("href", "")
            if not href or href in seen:
                continue
            title_span = a.find("span", {"title": True}) or a.find("span")
            title = (
                (title_span.get("title") or title_span.get_text(strip=True))
                if title_span
                else a.get_text(strip=True)
            )
            if title:
                seen.add(href)
                url = href if href.startswith("http") else f"{_INDEED_BASE}{href}"
                entries.append({"url": url, "title": title})

        # Fallback: any /viewjob link on the page
        if not entries:
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/viewjob" in href and href not in seen:
                    seen.add(href)
                    url = href if href.startswith("http") else f"{_INDEED_BASE}{href}"
                    entries.append({"url": url, "title": a.get_text(strip=True) or "Unknown"})

        # Debug safeguard: screenshot + page title when blocked or no results found
        if not entries:
            page_title = soup.find("title")
            print(f"[DEBUG] Indeed page title: {page_title.get_text(strip=True)!r}" if page_title else "[DEBUG] Indeed page title: (none)")
            dom_html = await search_page.content()
            with open("debug_indeed_dom.html", "w", encoding="utf-8") as f:
                f.write(dom_html)
            print("[DEBUG] Full DOM saved to debug_indeed_dom.html")
            await search_page.screenshot(path="debug_indeed.png", full_page=True)
            print("[DEBUG] Full-page screenshot saved to debug_indeed.png")
            await search_page.close()
            await browser.close()
            raise RuntimeError(
                "Indeed returned zero extractable jobs. Saved debug_indeed_dom.html and debug_indeed.png for inspection."
            )

        await search_page.close()

        # Concurrently scrape up to _MAX_JOBS job pages
        tasks = [_scrape_job_page(context, e["url"], e["title"]) for e in entries[:_MAX_JOBS]]
        scraped = await asyncio.gather(*tasks)

        await browser.close()

    valid = [j for j in scraped if len(j.get("description", "")) > 200]

    # Concurrently translate any Hebrew (or mixed) descriptions to English
    translate_tasks = [_translate_to_english(j) for j in valid]
    translated = await asyncio.gather(*translate_tasks)
    return list(translated)


def _run_async(coro):
    """Run an async coroutine from a synchronous context (Streamlit runs its own loop)."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── LLM helpers ────────────────────────────────────────────────────────────────

def _build_cv_profile(resume_text: str) -> str:
    """Distil a focused technical profile used as the FAISS query. Uses gpt-4o-mini (cheap)."""
    llm = ChatOpenAI(model=_MINI)
    prompt = f"""Extract a concise technical profile from this resume strictly for job-matching purposes.
Include ONLY:
1. Seniority level and years of experience
2. Core technical skills: ML/DL frameworks, languages, cloud tools
3. Domain expertise: NLP, computer vision, time-series, etc.
4. Nature of work: research, production pipelines, end-to-end, etc.

Max 150 words. Omit personal info, education details, and soft skills.

Resume:
{resume_text[:4000]}
"""
    return llm.invoke(prompt).content.strip()


class _JobClassification(BaseModel):
    role_type: str
    is_suitable: bool
    reason: str


def _classify_job(job_description: str, cv_profile: str) -> _JobClassification:
    """Gate-keep scraped jobs on required SKILLS, not job title. Uses gpt-4o-mini (cheap)."""
    llm = ChatOpenAI(model=_MINI).with_structured_output(_JobClassification)
    prompt = f"""You are a strict technical recruiter screening job postings for a senior machine-learning practitioner.

CRITICAL RULE: Ignore the job title completely. Judge solely on the day-to-day skills and tools
the posting actually requires. A posting titled "Data Scientist" that only asks for SQL and
dashboards is NOT suitable. A posting titled "Data Analyst" that requires building and deploying
predictive models IS suitable.

REJECT (is_suitable = False) when the primary day-to-day work is ANY of:
- BI / reporting tools: Tableau, Power BI, Looker, Qlik, Excel, Google Data Studio
- Writing SQL queries, building data pipelines, or ETL without ML modelling
- Describing, summarising, or visualising data (descriptive analytics)
- A/B test analysis, KPI tracking, or business dashboards
- No explicit mention of model building, training, or evaluation
- Requirements capped at 0-2 years of experience

ACCEPT (is_suitable = True) ONLY when the job explicitly and primarily requires:
- Classical ML: regression, classification, clustering, ensembles, feature engineering at scale
- Deep Learning: neural networks, transformers, PyTorch, TensorFlow, JAX
- Generative AI / LLMs: fine-tuning, RAG, prompt engineering in production, RLHF
- NLP: text modelling, embeddings, NER, summarisation, language models
- Computer vision: CNNs, object detection, image segmentation
- MLOps / model production: serving, monitoring, retraining pipelines, CI/CD for models
- Advanced probabilistic or statistical modelling (Bayesian inference, causal models, etc.)

When in doubt, REJECT. It is better to miss a borderline job than to let through a BI / analyst role.

Candidate profile (for seniority reference only — do not let it override the skill rules above):
{cv_profile}

Job description to evaluate:
{job_description[:2500]}

Return:
- role_type: honest short label based on actual required skills, e.g. "Senior DS", "ML Engineer",
  "NLP Researcher", "MLOps Engineer", "Data Analyst", "BI Analyst", "Junior DS"
- is_suitable: true / false
- reason: one sentence citing the specific skills (or lack thereof) that drove the decision
"""
    return llm.invoke(prompt)


class _JobScore(BaseModel):
    score: int       # 1-10 Scientific Rigor Score
    rationale: str   # one sentence explaining the score


def _score_job(job_description: str) -> _JobScore:
    """
    Assign a Scientific Rigor Score (1-10) based solely on required skills.
    Uses gpt-4o-mini — called on all FAISS candidates before the expensive gpt-4o step.
    """
    llm = ChatOpenAI(model=_MINI).with_structured_output(_JobScore)
    prompt = f"""You are a senior ML hiring manager. Assign a Scientific Rigor Score from 1 to 10
to this job posting based ONLY on the technical skills and day-to-day work it requires.

Scoring guide:
  8-10 (High — strong ML/DS role):
    - Core responsibility includes predictive modeling, classical machine learning, algorithm development,
      or advanced statistical modeling.
    - Building, training, evaluating, or improving ML / Deep Learning models (PyTorch, TensorFlow, JAX, scikit-learn).
    - End-to-end ML ownership (problem framing -> feature engineering -> modeling -> validation -> deployment/monitoring),
      even when described in high-level HR language.
    - Advanced modeling work (Bayesian methods, causal inference, time-series forecasting, optimization, experimentation).
    - Generative AI / LLM / NLP / computer vision work is a bonus, NOT a requirement for an 8+ score.
    - IMPORTANT: A legitimate Senior Data Scientist role focused on classical ML or statistical modeling should
      easily receive 8+ even if terms like "GenAI" or "LLMs" are never mentioned.

  5-7 (Medium — mixed or unclear):
    - Some modeling is present, but responsibilities are materially split with analytics/reporting/pipeline support.
    - Modeling expectations exist but ownership depth (training + evaluation + deployment) is unclear.
    - The posting is vague and does not clearly confirm end-to-end model development.

  1-4 (Low — pure analyst / BI only):
    - Apply this harsh band EXCLUSIVELY when the role is fundamentally analyst/BI/reporting work:
      SQL querying, dashboards, Tableau/Power BI/Looker/Qlik/Excel, KPI tracking, descriptive reporting.
    - No real expectation to build/train/evaluate ML models.
    - If the role includes actual model-building responsibilities, DO NOT place it in 1-4.

CRITICAL:
- Ignore the job title. Score strictly on day-to-day responsibilities.
- Job descriptions often use broad HR wording; infer intent from core responsibilities.
- If responsibilities imply building ML models end-to-end, assign a high score.
- Reserve 1-4 only for genuine non-ML analyst/BI roles.
- A posting titled "Data Scientist" that is dashboards-only must score 1-4.
- A posting titled "Data Analyst" that truly requires building predictive models can score 8-10.

Job description:
{job_description[:3000]}

Return:
- score: integer 1-10
- rationale: one sentence citing the specific skills or red flags that determined the score
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

def run_live_job_search(resume_text: str, progress_callback=None) -> tuple[list[dict], str]:
    """
    Optimised pipeline:
      1. GPT-4o Mini — distil CV profile (cheap triage)
      2. Playwright  — search il.indeed.com, extract job URLs
      3. Playwright  — scrape up to 15 job pages CONCURRENTLY (asyncio.gather)
      3b. GPT-4o Mini — translate Hebrew/mixed descriptions to English CONCURRENTLY
      4. GPT-4o Mini — classify each job, drop analyst/junior/BI roles (cheap triage)
      5. OpenAI      — build in-memory FAISS index on qualified jobs
      6. FAISS       — retrieve top 20 candidates by semantic similarity
      7. GPT-4o Mini — Scientific Rigor Score (1-10) for each candidate; sort; keep top 3
      8-10. GPT-4o  — deep gap analysis on top 3 scored jobs only (expensive, targeted)
    """
    TOTAL = 10
    step = 0

    def _progress(msg: str):
        nonlocal step
        step += 1
        if progress_callback:
            progress_callback(step, TOTAL, msg)

    # 1 — distil CV profile with gpt-4o-mini
    _progress("Extracting your technical profile with GPT-4o Mini...")
    cv_profile = _build_cv_profile(resume_text)

    # 2 — launch Playwright and scrape search page
    _progress("Launching Playwright browser, navigating to il.indeed.com...")
    raw_jobs = _run_async(_scrape_indeed("Senior Data Scientist", "Israel"))

    # 3 — concurrent scraping + translation already completed inside _scrape_indeed; report results
    _progress(f"Scraped and translated {len(raw_jobs)} job pages concurrently. Processing descriptions...")

    if not raw_jobs:
        raise RuntimeError(
            "Playwright found no job descriptions. "
            "Indeed may be blocking the scraper — try again in a few minutes."
        )

    # 4 — classify & filter with gpt-4o-mini
    _progress(f"Classifying {len(raw_jobs)} jobs with GPT-4o Mini (filtering analyst/junior roles)...")
    qualified: list[dict] = []
    for job in raw_jobs:
        try:
            clf = _classify_job(job["description"], cv_profile)
            if clf.is_suitable:
                job["role_type"] = clf.role_type
                qualified.append(job)
        except Exception:
            qualified.append(job)

    if len(qualified) < 3:
        qualified = raw_jobs  # safety net: filter was too aggressive

    # 5 — build FAISS index on qualified jobs only
    _progress(f"Building semantic FAISS index from {len(qualified)} qualified jobs...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(
        [j["description"] for j in qualified],
        embedding=embeddings,
        metadatas=[{"title": j["title"], "url": j["url"]} for j in qualified],
    )

    # 6 — retrieve broad candidate pool from FAISS (k=20 so scorer has real breadth)
    _progress("Retrieving top 20 candidates by semantic similarity...")
    candidates = vector_store.similarity_search(cv_profile, k=min(20, len(qualified)))

    # 7 — score each candidate with Scientific Rigor Score; keep only high-quality ML roles (>=8)
    _progress(f"Scoring {len(candidates)} candidates for scientific rigor with GPT-4o Mini...")
    scored: list[tuple[int, str, object]] = []   # (score, rationale, match doc)
    for match in candidates:
        try:
            s = _score_job(match.page_content)
            scored.append((s.score, s.rationale, match))
        except Exception:
            scored.append((5, "Scoring failed — defaulting to mid-range.", match))

    high_quality = [item for item in scored if item[0] >= 8]
    high_quality.sort(key=lambda x: x[0], reverse=True)
    top3 = high_quality[:3]

    if not top3:
        if progress_callback:
            progress_callback(TOTAL, TOTAL, "No high-quality ML roles found in this batch.")
        return [], cv_profile

    # 8-10 — deep gap analysis with gpt-4o (expensive — top 3 only)
    results: list[dict] = []
    for i, (score, rationale, match) in enumerate(top3, start=1):
        title = match.metadata.get("title", "Unknown")
        _progress(f"Deep gap analysis {i}/3 with GPT-4o: {title[:50]}...")
        analysis = _analyze_match(resume_text, match.page_content)
        results.append({
            "title": title,
            "url": match.metadata.get("url", ""),
            "score": score,
            "score_rationale": rationale,
            "matching_skills": analysis.matching_skills,
            "missing_skills": analysis.missing_skills,
            "summary": analysis.summary,
        })

    if progress_callback:
        progress_callback(TOTAL, TOTAL, "Done!")

    return results, cv_profile
