"""
ATS MCP Server

Exposes a single MCP tool:
  - extract_jobs_from_ats(url: str)

Given a Greenhouse or Comeet ATS URL, it fetches open/published jobs via public
HTTP endpoints and returns a clean JSON string of job postings:
  { title, location, description, url }
"""

from __future__ import annotations

import html as html_lib
import json
import re
import sys
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ATS Job Extractor")

_USER_AGENT = (
    "CareerNavigator/1.0 (+https://github.com/your-repo) "
    "python-httpx"
)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_html_to_text(text: str, *, max_chars: int = 8000) -> str:
    """Best-effort conversion of HTML-ish text into plain text."""
    if not text:
        return ""

    # Greenhouse returns HTML entities; Comeet often returns plain text but we keep this robust.
    text = html_lib.unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _detect_ats(url: str) -> Literal["greenhouse", "comeet"]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()

    if "greenhouse" in host or "boards-api.greenhouse.io" in host or "greenhouse.io" in path:
        return "greenhouse"
    if "comeet" in host or "comeet" in path:
        return "comeet"

    raise ValueError(f"Unsupported ATS provider for url={url!r}")


def _greenhouse_board_token(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    parts = [p for p in (parsed.path or "").split("/") if p]

    # Common forms:
    #  - https://boards.greenhouse.io/<board_token>/jobs
    #  - https://boards.greenhouse.io/<board_token>/jobs/<job_id>
    #  - https://<board_token>.greenhouse.io/jobs
    if host == "boards.greenhouse.io":
        if not parts:
            raise ValueError("Greenhouse URL is missing board token path segment.")
        return parts[0]

    if host.endswith(".greenhouse.io"):
        # <board_token>.greenhouse.io
        return host.split(".")[0]

    # Fallback: try first path segment.
    if parts:
        return parts[0]

    raise ValueError("Could not extract Greenhouse board token from URL.")


async def _fetch_greenhouse_jobs(url: str) -> list[dict[str, Any]]:
    board_token = _greenhouse_board_token(url)
    endpoint = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        resp = await client.get(endpoint, params={"content": "true"})
        resp.raise_for_status()
        payload = resp.json()

    jobs = payload.get("jobs", []) or []
    out: list[dict[str, Any]] = []
    for j in jobs:
        title = j.get("title") or ""
        loc = j.get("location") or {}
        location = loc.get("name") if isinstance(loc, dict) else ""
        description = _clean_html_to_text(j.get("content", "") or "")
        job_url = j.get("absolute_url") or ""

        out.append(
            {
                "title": title,
                "location": location or "",
                "description": description,
                "url": job_url,
            }
        )
    return out


def _extract_comeet_identifiers_from_query(parsed) -> tuple[str | None, str | None]:
    query = parse_qs(parsed.query or "")
    token = (query.get("token") or [None])[0]

    # Some Comeet integrations may pass company uid directly in query as well.
    company_uid = (
        (query.get("company_uid") or query.get("company-uid") or [None])[0]
    )
    return company_uid, token


def _extract_comeet_identifiers_from_path(parsed) -> tuple[str | None, str | None]:
    parts = [p for p in (parsed.path or "").split("/") if p]

    # /careers-api/2.0/company/{company_uid}/positions
    if "company" in parts:
        idx = parts.index("company")
        if idx + 1 < len(parts):
            company_uid = parts[idx + 1]
            return company_uid, None

    # Standard hosted board URL: /jobs/{company_uid}/{token}
    # e.g. https://www.comeet.com/jobs/audiocodes/85.004
    if len(parts) >= 3 and parts[0] == "jobs":
        return parts[1], parts[2]

    return None, None


def _extract_comeet_identifiers_from_html(html: str) -> tuple[str | None, str | None]:
    """
    Best-effort extraction from Comeet's embed-config style JavaScript.

    Typical snippet:
      COMEET.init({ "token": "...", "company-uid": "..." })
    """

    # Cover both single and double quotes.
    token_re = re.compile(r"['\"]token['\"]\s*:\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
    cuid_re = re.compile(
        r"['\"]company-uid['\"]\s*:\s*['\"]([^'\"]+)['\"]", re.IGNORECASE
    )
    token_match = token_re.search(html)
    cuid_match = cuid_re.search(html)
    token = token_match.group(1) if token_match else None
    company_uid = cuid_match.group(1) if cuid_match else None
    return company_uid, token


async def _fetch_comeet_jobs(url: str) -> list[dict[str, Any]]:
    parsed = urlparse(url)

    company_uid, token = _extract_comeet_identifiers_from_query(parsed)
    if not company_uid:
        company_uid, token_from_path = _extract_comeet_identifiers_from_path(parsed)
        token = token or token_from_path

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        if not company_uid or not token:
            # Fetch the provided URL and look for COMEET.init config.
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text
            extracted_cuid, extracted_token = _extract_comeet_identifiers_from_html(html)
            company_uid = company_uid or extracted_cuid
            token = token or extracted_token

        if not company_uid or not token:
            raise ValueError(
                "Could not extract Comeet company identifiers (company_uid/token). "
                "Pass a Comeet URL that contains them in query/path, "
                "or ensure the board page embeds COMEET.init config."
            )

        endpoint = f"https://www.comeet.co/careers-api/2.0/company/{company_uid}/positions"
        resp = await client.get(
            endpoint,
            params={"token": token, "details": "true"},
        )
        resp.raise_for_status()
        positions = resp.json() or []

    out: list[dict[str, Any]] = []
    for pos in positions:
        title = pos.get("name") or ""
        loc = pos.get("location") or {}
        location = loc.get("name") if isinstance(loc, dict) else ""

        details = pos.get("details") or []
        description_parts: list[str] = []
        requirements_parts: list[str] = []

        # Comeet returns details as a list like:
        # [{ "name": "Description", "value": "..."} , { "name":"Requirements", "value":"..."}]
        if isinstance(details, list):
            for d in details:
                if not isinstance(d, dict):
                    continue
                name = (d.get("name") or "").strip().lower()
                value = d.get("value") or ""
                if not value:
                    continue

                if name in {"description", "about", "job description", "role"}:
                    description_parts.append(_clean_html_to_text(str(value)))
                elif name in {"requirements", "what we are looking for", "qualifications", "must have"}:
                    requirements_parts.append(_clean_html_to_text(str(value)))

        description = ""
        if requirements_parts:
            description = "\n\n".join([p for p in requirements_parts if p]).strip()
        elif description_parts:
            description = "\n\n".join([p for p in description_parts if p]).strip()
        else:
            # Some Comeet boards may not include details even with details=true; fall back to empty.
            description = ""

        job_url = (
            pos.get("url_active_page")
            or pos.get("url_comeet_hosted_page")
            or pos.get("url_recruit_hosted_page")
            or ""
        )

        out.append(
            {
                "title": title,
                "location": location or "",
                "description": description,
                "url": job_url,
            }
        )

    return out


@mcp.tool()
async def extract_jobs_from_ats(url: str) -> str:
    """
    Extract open job postings from a public ATS board URL.

    Args:
        url: Greenhouse or Comeet board URL.

    Returns:
        A JSON string (not an array) representing a list of jobs:
          [{"title":..., "location":..., "description":..., "url":...}, ...]
    """
    ats = _detect_ats(url)

    if ats == "greenhouse":
        jobs = await _fetch_greenhouse_jobs(url)
    else:
        jobs = await _fetch_comeet_jobs(url)

    # Always return JSON string to keep the MCP tool contract stable.
    return json.dumps(jobs, ensure_ascii=False)


if __name__ == "__main__":
    # stdio is ideal for local dev: the MCP client launches this as a subprocess.
    transport = "stdio"
    if len(sys.argv) > 1 and sys.argv[1] in {"stdio", "http", "streamable-http"}:
        transport = sys.argv[1]
    mcp.run(transport=transport)

