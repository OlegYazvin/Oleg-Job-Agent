from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from html import unescape
import json
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
import httpx
from pydantic import BaseModel, Field


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
AI_EVIDENCE_PATTERNS = (
    "artificial intelligence",
    "machine learning",
    "generative ai",
    "agentic",
    "llm",
    "chatbot",
    "chatbots",
    "conversational ai",
    "computer vision",
    "natural language processing",
    "nlp",
    "predictive model",
    "model orchestration",
)

ATS_HOST_MAP = {
    "greenhouse.io": "Greenhouse",
    "job-boards.greenhouse.io": "Greenhouse",
    "boards.greenhouse.io": "Greenhouse",
    "jobs.lever.co": "Lever",
    "ashbyhq.com": "Ashby",
    "myworkdayjobs.com": "Workday",
    "careers.workday.com": "Workday",
    "icims.com": "iCIMS",
    "jobvite.com": "Jobvite",
    "smartrecruiters.com": "SmartRecruiters",
    "recruitee.com": "Recruitee",
    "careers.tellent.com": "Recruitee",
    "jobscore.com": "JobScore",
    "portal.dynamicsats.com": "DynamicsATS",
}


class JobPageSnapshot(BaseModel):
    requested_url: str
    resolved_url: str
    ats_platform: str
    status_code: int
    page_title: str | None = None
    company_name: str | None = None
    role_title: str | None = None
    location_text: str | None = None
    is_fully_remote: bool | None = None
    posted_date_iso: str | None = None
    posted_date_text: str | None = None
    base_salary_min_usd: int | None = None
    base_salary_max_usd: int | None = None
    salary_text: str | None = None
    text_excerpt: str = ""
    evidence_snippets: list[str] = Field(default_factory=list)


async def fetch_job_page(url: str) -> JobPageSnapshot:
    last_error: httpx.RequestError | None = None
    response: httpx.Response | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                timeout=20.0,
            ) as client:
                response = await client.get(url)
            break
        except httpx.RequestError as exc:
            last_error = exc
            if attempt < 2:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

    if response is None:
        error_text = str(last_error) if last_error is not None else "Unknown fetch error."
        return JobPageSnapshot(
            requested_url=url,
            resolved_url=url,
            ats_platform=_detect_ats_platform(url),
            status_code=0,
            text_excerpt=error_text[:7000],
            evidence_snippets=[f"Fetch error: {error_text}"],
        )

    html = response.text
    soup = BeautifulSoup(html, "html.parser")
    resolved_url = str(response.url)
    ats_platform = _detect_ats_platform(resolved_url)
    plain_text = " ".join(soup.stripped_strings)
    json_ld = _extract_job_posting_json_ld(soup)
    structured_description_text = _extract_jsonld_description_text(json_ld) if json_ld else ""
    combined_text = " ".join(part for part in (plain_text, structured_description_text) if part)

    snapshot = JobPageSnapshot(
        requested_url=url,
        resolved_url=resolved_url,
        ats_platform=ats_platform,
        status_code=response.status_code,
        page_title=soup.title.get_text(strip=True) if soup.title else None,
        text_excerpt=combined_text[:7000],
    )

    evidence: list[str] = []
    if json_ld:
        _merge(snapshot, _extract_generic_jobposting_fields(json_ld, combined_text, evidence))

    greenhouse_api_ref = _extract_greenhouse_board_job_reference(url, html)
    if greenhouse_api_ref:
        board_token, job_id = greenhouse_api_ref
        api_payload = await _fetch_greenhouse_job_api_payload(board_token, job_id)
        if api_payload:
            _merge(snapshot, _extract_greenhouse_api_fields(api_payload, evidence))

    if ats_platform == "Greenhouse":
        _merge(snapshot, _extract_greenhouse_fields(html, evidence))
    elif ats_platform == "Lever":
        _merge(snapshot, _extract_lever_fields(soup, html, evidence))
    elif ats_platform == "Ashby":
        _merge(snapshot, _extract_ashby_fields(html, evidence))
    elif ats_platform == "Recruitee":
        _merge(snapshot, _extract_recruitee_fields(html, evidence))
    elif ats_platform == "JobScore":
        _merge(snapshot, _extract_jobscore_fields(soup, html, evidence))

    # Generic fallbacks after ATS-specific extraction.
    salary_min, salary_max, salary_text = _extract_salary_range(combined_text)
    if snapshot.base_salary_min_usd is None and salary_min is not None:
        snapshot.base_salary_min_usd = salary_min
        snapshot.base_salary_max_usd = salary_max
        snapshot.salary_text = snapshot.salary_text or salary_text
        evidence.append(f"Salary text: {salary_text}")

    if not snapshot.posted_date_iso:
        generic_posted = _extract_relative_posted_text(combined_text)
        if generic_posted:
            snapshot.posted_date_text = generic_posted
            evidence.append(f"Posted text: {generic_posted}")

    if not snapshot.location_text:
        location = _extract_location_fallback(combined_text)
        if location:
            snapshot.location_text = location
            evidence.append(f"Location text: {location}")
            if snapshot.is_fully_remote is False and _location_text_is_explicitly_remote(location):
                remote_override = _infer_remote_status(location, combined_text)
                if remote_override:
                    snapshot.is_fully_remote = True
                    evidence.append("Remote evidence found after location fallback.")

    if snapshot.is_fully_remote is None:
        snapshot.is_fully_remote = _infer_remote_status(snapshot.location_text, combined_text)
        if snapshot.is_fully_remote:
            evidence.append("Remote evidence found in the job page text.")

    if not snapshot.posted_date_text and snapshot.posted_date_iso:
        snapshot.posted_date_text = snapshot.posted_date_iso

    if snapshot.page_title and not snapshot.role_title:
        snapshot.role_title = _title_to_role(snapshot.page_title)

    ai_evidence = _extract_ai_context_snippets(combined_text)
    snapshot.evidence_snippets = list(dict.fromkeys([*ai_evidence, *evidence]))[:8]
    return snapshot


def _merge(snapshot: JobPageSnapshot, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        current = getattr(snapshot, key)
        if current in (None, "", []):
            setattr(snapshot, key, value)


def _detect_ats_platform(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    for fragment, label in ATS_HOST_MAP.items():
        if fragment in host:
            return label
    return host or "Unknown"


def _extract_job_posting_json_ld(soup: BeautifulSoup) -> dict[str, Any] | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        job_posting = _find_job_posting_node(payload)
        if job_posting:
            return job_posting
    return None


def _extract_json_object_after_marker(html: str, marker: str) -> dict[str, Any] | None:
    marker_index = html.find(marker)
    if marker_index == -1:
        return None
    start = html.find("{", marker_index + len(marker))
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(html)):
        char = html[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = html[start : index + 1]
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _extract_escaped_json_object_after_marker(html: str, marker: str) -> dict[str, Any] | None:
    marker_index = html.find(marker)
    if marker_index == -1:
        return None
    start = html.find("{", marker_index + len(marker))
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(html)):
        char = html[index]
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw_candidate = html[start : index + 1]
                try:
                    payload = json.loads(unescape(raw_candidate))
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _find_job_posting_node(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        type_value = payload.get("@type")
        if type_value == "JobPosting" or (isinstance(type_value, list) and "JobPosting" in type_value):
            return payload
        for value in payload.values():
            found = _find_job_posting_node(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _find_job_posting_node(item)
            if found:
                return found
    return None


def _extract_generic_jobposting_fields(
    payload: dict[str, Any], plain_text: str, evidence: list[str]
) -> dict[str, Any]:
    company = _nested_get(payload, "hiringOrganization", "name")
    title = payload.get("title")
    location = _extract_jsonld_location(payload)
    salary_min, salary_max, salary_text = _extract_jsonld_salary(payload)
    date_posted = payload.get("datePosted")
    remote_context = " ".join(part for part in (str(title or ""), plain_text) if part)
    remote = _infer_remote_status(location, remote_context)

    values = {
        "company_name": company,
        "role_title": title,
        "location_text": location,
        "base_salary_min_usd": salary_min,
        "base_salary_max_usd": salary_max,
        "salary_text": salary_text,
        "posted_date_iso": _normalize_iso_date(date_posted),
        "posted_date_text": _normalize_iso_date(date_posted),
        "is_fully_remote": remote,
    }

    if company:
        evidence.append(f"Company from structured data: {company}")
    if title:
        evidence.append(f"Title from structured data: {title}")
    if date_posted:
        evidence.append(f"datePosted from structured data: {date_posted}")
    return values


def _extract_ashby_fields(html: str, evidence: list[str]) -> dict[str, Any]:
    payload = _extract_json_object_after_marker(html, "window.__appData = ")
    if not payload:
        return {}
    posting = payload.get("posting") if isinstance(payload.get("posting"), dict) else {}
    organization = payload.get("organization") if isinstance(payload.get("organization"), dict) else {}
    description_html = str(posting.get("descriptionHtml") or "")
    description_text = _html_to_text(description_html)
    structured_text = " ".join(part for part in (description_text, str(posting.get("locationName") or "")) if part)
    salary_min, salary_max, salary_text = _extract_salary_range(description_text or structured_text)
    location = str(posting.get("locationName") or posting.get("locationExternalName") or "").strip() or None
    published_at = posting.get("publishedAt") or posting.get("publishedDate")
    values = {
        "company_name": str(organization.get("name") or "").strip() or None,
        "role_title": str(posting.get("title") or "").strip() or None,
        "location_text": location,
        "is_fully_remote": _infer_remote_status(location, structured_text),
        "posted_date_iso": _normalize_iso_date(str(published_at or "")) if published_at else None,
        "posted_date_text": _normalize_iso_date(str(published_at or "")) if published_at else None,
        "base_salary_min_usd": salary_min,
        "base_salary_max_usd": salary_max,
        "salary_text": salary_text,
    }
    if values["company_name"]:
        evidence.append(f"Ashby organization: {values['company_name']}")
    if values["role_title"]:
        evidence.append(f"Ashby posting title: {values['role_title']}")
    if values["location_text"]:
        evidence.append(f"Ashby location: {values['location_text']}")
    return values


def _extract_recruitee_fields(html: str, evidence: list[str]) -> dict[str, Any]:
    offer = _extract_escaped_json_object_after_marker(html, "&quot;offers&quot;:[")
    if not offer:
        return {}
    translations = offer.get("translations") if isinstance(offer.get("translations"), dict) else {}
    english = translations.get("en") if isinstance(translations.get("en"), dict) else {}
    title = str(english.get("name") or english.get("title") or "").strip() or None
    country = str(english.get("country") or offer.get("countryCode") or "").strip()
    city = str(offer.get("city") or "").strip()
    location_parts = [part for part in (city, country) if part]
    salary = offer.get("salary") if isinstance(offer.get("salary"), dict) else {}
    salary_currency = str(salary.get("currency") or "").upper()
    salary_min = salary_max = None
    salary_text = None
    if salary:
        min_raw = salary.get("min")
        max_raw = salary.get("max")
        period = str(salary.get("period") or "year").lower()
        salary_text = f"{salary_currency} {min_raw:g} - {max_raw:g} per {period}" if isinstance(min_raw, (int, float)) and isinstance(max_raw, (int, float)) else None
        if salary_currency in {"USD", "US$", "$"}:
            if isinstance(min_raw, (int, float)):
                salary_min = int(min_raw)
            if isinstance(max_raw, (int, float)):
                salary_max = int(max_raw)
            salary_text = (
                f"${salary_min:,} - ${salary_max:,} per {period}"
                if salary_min is not None and salary_max is not None
                else salary_text
            )
    description_html = str(english.get("descriptionHtml") or "")
    values = {
        "role_title": title,
        "location_text": ", ".join(location_parts) or None,
        "is_fully_remote": True if offer.get("remote") is True else (False if offer.get("hybrid") or offer.get("onSite") else _infer_remote_status(", ".join(location_parts), _html_to_text(description_html))),
        "base_salary_min_usd": salary_min,
        "base_salary_max_usd": salary_max,
        "salary_text": salary_text,
    }
    if values["role_title"]:
        evidence.append(f"Recruitee title: {values['role_title']}")
    if values["location_text"]:
        evidence.append(f"Recruitee location: {values['location_text']}")
    if values["salary_text"]:
        evidence.append(f"Recruitee salary: {values['salary_text']}")
    return values


def _extract_jobscore_fields(soup: BeautifulSoup, html: str, evidence: list[str]) -> dict[str, Any]:
    meta_by_key: dict[str, str] = {}
    for meta in soup.select("meta[name], meta[property]"):
        key = (meta.get("name") or meta.get("property") or "").strip().lower()
        content = (meta.get("content") or "").strip()
        if key and content:
            meta_by_key[key] = content
    title = meta_by_key.get("title") or meta_by_key.get("twitter:title") or (soup.title.get_text(strip=True) if soup.title else "")
    description = meta_by_key.get("description") or meta_by_key.get("twitter:description") or ""
    match = re.search(
        r"Share the\s+(?P<role>.+?)\s+open at\s+(?P<company>.+?)\s+in\s+(?P<location>.+?)(?:, powered by JobScore|[.])",
        title,
        re.I,
    )
    role_title = company_name = location_text = None
    if match:
        role_title = match.group("role").strip()
        company_name = match.group("company").strip()
        location_text = match.group("location").strip().rstrip(".")
    body_text = " ".join(part for part in (title, description, _html_to_text(html)) if part)
    values = {
        "company_name": company_name,
        "role_title": role_title,
        "location_text": location_text,
        "is_fully_remote": _infer_remote_status(location_text, body_text),
    }
    if company_name:
        evidence.append(f"JobScore company: {company_name}")
    if role_title:
        evidence.append(f"JobScore title: {role_title}")
    if location_text:
        evidence.append(f"JobScore location: {location_text}")
    return values


def _extract_jsonld_description_text(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    raw_description = payload.get("description")
    if not isinstance(raw_description, str):
        return ""
    return _html_to_text(unescape(raw_description))


def _extract_greenhouse_fields(html: str, evidence: list[str]) -> dict[str, Any]:
    payload = _extract_greenhouse_remix_context(html)
    if not payload:
        return {}

    pay_ranges = payload.get("pay_ranges") or []
    salary_min = salary_max = None
    salary_text = None
    if pay_ranges:
        first_range = pay_ranges[0]
        salary_min = _parse_money(first_range.get("min"))
        salary_max = _parse_money(first_range.get("max"))
        if first_range.get("min") and first_range.get("max"):
            salary_text = f"{first_range.get('min')} - {first_range.get('max')}"
        if salary_text:
            evidence.append(f"Greenhouse pay range: {salary_text}")

    published_at = payload.get("published_at")
    job_location = payload.get("job_post_location")
    if published_at:
        evidence.append(f"Greenhouse published_at: {published_at}")
    if job_location:
        evidence.append(f"Greenhouse location: {job_location}")

    return {
        "company_name": payload.get("company_name"),
        "role_title": payload.get("title"),
        "location_text": job_location,
        "posted_date_iso": _normalize_iso_date(published_at),
        "posted_date_text": _normalize_iso_date(published_at),
        "salary_text": salary_text,
        "base_salary_min_usd": salary_min,
        "base_salary_max_usd": salary_max,
        "is_fully_remote": _infer_remote_status(job_location, _html_to_text(payload.get("content", ""))),
    }


def _extract_greenhouse_board_job_reference(url: str, html: str) -> tuple[str, str] | None:
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    board = (query_params.get("board") or [None])[0]
    gh_jid = (query_params.get("gh_jid") or [None])[0]
    if board and gh_jid:
        return board, gh_jid

    host = (parsed.netloc or "").lower()
    segments = [segment for segment in parsed.path.strip("/").split("/") if segment]
    if "greenhouse.io" in host and "jobs" in segments:
        try:
            jobs_index = segments.index("jobs")
        except ValueError:
            jobs_index = -1
        if jobs_index > 0 and len(segments) > jobs_index + 1:
            return segments[jobs_index - 1], segments[jobs_index + 1]

    html_unescaped = unescape(html)
    script_match = re.search(r"boards\.greenhouse\.io/embed/job_board/js\?for=([a-z0-9_-]+)", html_unescaped, re.I)
    if script_match and gh_jid:
        return script_match.group(1), gh_jid

    return None


async def _fetch_greenhouse_job_api_payload(board_token: str, job_id: str) -> dict[str, Any] | None:
    api_url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs/{job_id}"
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
            timeout=20.0,
        ) as client:
            response = await client.get(api_url)
    except httpx.RequestError:
        return None
    if response.status_code != 200:
        return None
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_greenhouse_api_fields(payload: dict[str, Any], evidence: list[str]) -> dict[str, Any]:
    company = payload.get("company_name")
    title = payload.get("title")
    location = _nested_get(payload, "location", "name")
    first_published = payload.get("first_published") or payload.get("updated_at")
    content_html = payload.get("content") or ""
    content_text = _html_to_text(unescape(content_html))
    salary_min, salary_max, salary_text = _extract_salary_range(content_text)

    if company:
        evidence.append(f"Greenhouse API company: {company}")
    if title:
        evidence.append(f"Greenhouse API title: {title}")
    evidence.extend(_extract_ai_context_snippets(content_text)[:2])
    if first_published:
        evidence.append(f"Greenhouse API first_published: {first_published}")
    if location:
        evidence.append(f"Greenhouse API location: {location}")
    if salary_text:
        evidence.append(f"Salary text: {salary_text}")

    return {
        "company_name": company,
        "role_title": title,
        "location_text": location,
        "posted_date_iso": _normalize_iso_date(first_published),
        "posted_date_text": _normalize_iso_date(first_published),
        "base_salary_min_usd": salary_min,
        "base_salary_max_usd": salary_max,
        "salary_text": salary_text,
        "is_fully_remote": _infer_remote_status(location, content_text),
    }


def _extract_greenhouse_remix_context(html: str) -> dict[str, Any] | None:
    match = re.search(r"window\.__remixContext\s*=\s*(\{.*?\});\s*</script>", html, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
        return payload["state"]["loaderData"]["routes/$url_token_.jobs_.$job_post_id"]["jobPost"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _extract_lever_fields(soup: BeautifulSoup, html: str, evidence: list[str]) -> dict[str, Any]:
    title = soup.title.get_text(strip=True) if soup.title else None
    location = None
    location_el = soup.select_one(".posting-categories .location, .posting-categories .sort-by-location")
    if location_el:
        location = location_el.get_text(" ", strip=True)

    workplace_el = soup.select_one(".posting-categories .workplaceTypes")
    workplace_text = workplace_el.get_text(" ", strip=True) if workplace_el else ""

    salary_el = soup.select_one('[data-qa="salary-range"]')
    salary_min = salary_max = None
    salary_text = None
    if salary_el:
        salary_min, salary_max, salary_text = _extract_salary_range(salary_el.get_text(" ", strip=True))
        if salary_text:
            evidence.append(f"Lever salary range: {salary_text}")

    posted_match = re.search(r'"datePosted"\s*:\s*"([^"]+)"', html)
    if posted_match:
        evidence.append(f"Lever datePosted: {posted_match.group(1)}")

    company_name, role_title = _split_company_and_role(title)
    combined_text = " ".join(part for part in (location, workplace_text) if part)
    plain_text = _html_to_text(html)
    remote_context = " ".join(part for part in (title or "", combined_text, plain_text[:4000]) if part)
    return {
        "company_name": company_name,
        "role_title": role_title,
        "location_text": combined_text or location,
        "posted_date_iso": _normalize_iso_date(posted_match.group(1) if posted_match else None),
        "posted_date_text": _normalize_iso_date(posted_match.group(1) if posted_match else None),
        "salary_text": salary_text,
        "base_salary_min_usd": salary_min,
        "base_salary_max_usd": salary_max,
        "is_fully_remote": _infer_remote_status(combined_text or location, remote_context),
    }


def _extract_jsonld_salary(payload: dict[str, Any]) -> tuple[int | None, int | None, str | None]:
    base_salary = payload.get("baseSalary")
    if not isinstance(base_salary, dict):
        return None, None, None

    value = base_salary.get("value")
    currency = base_salary.get("currency") or "USD"
    if isinstance(value, dict):
        min_value = value.get("minValue")
        max_value = value.get("maxValue")
    else:
        min_value = value
        max_value = value

    salary_min = _parse_money(min_value)
    salary_max = _parse_money(max_value)
    if salary_min is None and salary_max is None:
        return None, None, None

    if salary_min is not None and salary_max is not None and currency == "USD":
        return salary_min, salary_max, f"${salary_min:,} - ${salary_max:,}"
    if salary_min is not None and currency == "USD":
        return salary_min, salary_min, f"${salary_min:,}"
    return salary_min, salary_max, None


def _extract_jsonld_location(payload: dict[str, Any]) -> str | None:
    job_location = payload.get("jobLocation")
    if isinstance(job_location, list) and job_location:
        job_location = job_location[0]
    if isinstance(job_location, dict):
        address = job_location.get("address")
        if isinstance(address, dict):
            parts = [
                address.get("addressLocality"),
                address.get("addressRegion"),
                _jsonld_address_country_name(address.get("addressCountry")),
            ]
            return ", ".join(part for part in parts if part) or None
    if payload.get("jobLocationType") == "TELECOMMUTE":
        return "Remote"
    return None


def _jsonld_address_country_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("name") or value.get("value")
    return None


def _extract_ai_context_snippets(text: str) -> list[str]:
    if not text:
        return []
    normalized = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    snippets: list[str] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(pattern in lowered for pattern in AI_EVIDENCE_PATTERNS) or re.search(r"\b(ai|ml)\b", lowered):
            cleaned = sentence.strip()
            if cleaned:
                snippets.append(f"AI context: {cleaned[:220]}")
        if len(snippets) >= 3:
            break
    return snippets


def _extract_salary_range(text: str) -> tuple[int | None, int | None, str | None]:
    range_pattern = re.compile(
        r"(?P<min>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)\s*(?:-|to)\s*(?P<max>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)"
    )
    match = range_pattern.search(text)
    if not match:
        # Handle explicit single-value salary statements like:
        # "base salary ... up to $180,000" or "base salary ... $210,000"
        single_patterns = (
            re.compile(
                r"(?:annual\s+)?base\s+salary[^$]{0,120}?(?:up to|maximum(?:\s+of)?|max(?:imum)?(?:\s+of)?)\s*(?P<value>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)",
                re.I,
            ),
            re.compile(
                r"(?:annual\s+)?base\s+salary[^$]{0,120}(?P<value>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)",
                re.I,
            ),
            re.compile(
                r"(?:salary|compensation|pay range)[^$]{0,60}(?P<value>\$?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?)",
                re.I,
            ),
        )
        for pattern in single_patterns:
            single_match = pattern.search(text)
            if not single_match:
                continue
            value_raw = single_match.group("value")
            parsed = _parse_money(value_raw)
            if parsed is None:
                continue
            return parsed, parsed, value_raw.replace("  ", " ").strip()
        return None, None, None

    min_value = _parse_money(match.group("min"))
    max_value = _parse_money(match.group("max"))
    if (
        min_value is None
        or max_value is None
        or not _money_token_looks_salary(match.group("min"), min_value)
        or not _money_token_looks_salary(match.group("max"), max_value)
    ):
        return None, None, None
    return min_value, max_value, match.group(0).replace("  ", " ").strip()


def _money_token_looks_salary(value: str, parsed_value: int | None = None) -> bool:
    lowered = value.lower().strip()
    if "$" in lowered or lowered.endswith(("k", "m")):
        return True
    numeric_value = parsed_value if parsed_value is not None else _parse_money(value)
    return numeric_value is not None and numeric_value >= 30_000


def _extract_relative_posted_text(text: str) -> str | None:
    match = re.search(
        r"(posted\s+)?(?P<value>today|yesterday|\d+\s+(?:day|days|week|weeks)\s+ago)",
        text,
        re.I,
    )
    if match:
        return match.group("value")
    return None


def _extract_location_fallback(text: str) -> str | None:
    if re.search(r"\b(required\s+in[-\s]*office|in[-\s]*office|on[-\s]*site|onsite)\b", text, re.I):
        return "In Office"
    match = re.search(r"\b(remote|anywhere in the world|united states|worldwide)\b", text, re.I)
    return match.group(1) if match else None


def _infer_remote_status(location_text: str | None, text: str) -> bool | None:
    haystacks = " ".join(part for part in (location_text or "", text or "") if part).lower()
    if not haystacks.strip():
        return None
    if re.search(r"\bhybrid if located\b.*\bor remote\b", haystacks):
        return True
    if re.search(r"\bor remote with travel\b", haystacks):
        return True
    if re.search(r"\bwork persona\s*:\s*required in[-\s]*office\b", haystacks):
        return False
    if re.search(r"\bwork persona\s*:\s*remote\b", haystacks):
        return True
    strong_remote_signal = any(
        token in haystacks
        for token in (
            "fully remote",
            "100% remote",
            "remote-first",
            "remote only",
            "remote-only",
            "remote role",
            "remote opportunity",
            "remote position",
            "full-time remote",
            "full time remote",
            "remote in",
            "remote within",
            "remote -",
            "remote,",
            "/ remote",
            "remote /",
            "(remote)",
            " remotely ",
            "remotely in",
            "remotely within",
            "remotely from",
            "work from anywhere",
            "work from home",
            "telecommute",
            "telecommuting",
            "virtual role",
            "virtual position",
            "anywhere in the world",
        )
    )
    if strong_remote_signal and not any(
        token in haystacks
        for token in (
            "required in office",
            "required in-office",
            "in office",
            "in-office",
            "office-based",
            "hybrid",
            "on-site",
            "onsite",
            "office-first",
            "must be based in office",
            "work from office",
        )
    ):
        return True
    if _location_text_is_specific_non_remote(location_text):
        return False
    if _location_text_is_explicitly_remote(location_text):
        return True
    if (
        "required in office" in haystacks
        or "required in-office" in haystacks
        or "in office" in haystacks
        or "in-office" in haystacks
        or "office-based" in haystacks
    ):
        return False
    if any(
        token in haystacks
        for token in (
            "hybrid",
            "on-site",
            "onsite",
            "office-first",
            "must be based in office",
            "work from office",
        )
    ):
        return False
    if any(
        token in haystacks
        for token in (
            "fully remote",
            "100% remote",
            "remote-first",
            "remote only",
            "remote-only",
            "remote position",
            "remote role",
            "remote eligible",
            "remote opportunity",
            "remote within",
            "remote in",
            "remote -",
            "remote,",
            "/ remote",
            "remote /",
            "work from anywhere",
            "work from home",
            "telecommute",
            "telecommuting",
            "virtual role",
            "virtual position",
            "anywhere in the world",
        )
    ):
        return True
    if "remote" in haystacks:
        return True
    return None


def _location_text_is_explicitly_remote(location_text: str | None) -> bool:
    lowered = (location_text or "").lower().strip()
    if not lowered:
        return False
    return any(
        token in lowered
        for token in (
            "remote",
            "virtual",
            "telecommute",
            "telecommuting",
            "work from home",
            "work from anywhere",
        )
    )


def _location_text_is_specific_non_remote(location_text: str | None) -> bool:
    lowered = (location_text or "").lower().strip()
    if not lowered or _location_text_is_explicitly_remote(lowered):
        return False
    if any(
        token in lowered
        for token in (
            "required in office",
            "required in-office",
            "in office",
            "in-office",
            "hybrid",
            "on-site",
            "onsite",
            "office",
        )
    ):
        return True

    generic_locations = {
        "united states",
        "united states of america",
        "usa",
        "us",
        "worldwide",
        "global",
    }
    if lowered in generic_locations:
        return False

    if re.search(r"\b[a-z0-9 .&'-]+,\s*[a-z]{2}\b", lowered):
        return True
    if re.search(r"\b[a-z0-9 .&'-]+,\s*[a-z ]+\b", lowered):
        return True
    if re.search(r"\b\d{2,}\s+[a-z0-9 .'-]+", lowered):
        return True
    return False


def _normalize_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        if "T" in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat()
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return None


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _parse_money(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)

    raw = str(value).strip().lower().replace(",", "").replace("$", "")
    multiplier = 1
    if raw.endswith("k"):
        multiplier = 1_000
        raw = raw[:-1]
    elif raw.endswith("m"):
        multiplier = 1_000_000
        raw = raw[:-1]
    try:
        return int(float(raw) * multiplier)
    except ValueError:
        return None


def _split_company_and_role(title: str | None) -> tuple[str | None, str | None]:
    if not title:
        return None, None
    if " - " not in title:
        return None, _title_to_role(title)
    company, role = title.split(" - ", 1)
    return company.strip(), role.strip()


def _title_to_role(title: str) -> str:
    return title.replace("Job Application for ", "").replace(" at ", " | ").split(" | ")[0].strip()


def _html_to_text(html: str) -> str:
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
