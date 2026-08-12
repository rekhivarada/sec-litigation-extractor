"""
pipeline_core.py — importable extraction logic, shared by the CLI script and the Streamlit UI.
Same accuracy fixes as pipeline_full.py: --year filter, junk-row filtering, currency-aware
amounts, evidence-quote grounding checks.
"""

import re
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
from bs4 import BeautifulSoup

SEC_EMAIL       = "varadarekhi@gmail.com"   # EDIT: your real email (required by SEC)
OLLAMA_MODEL    = "qwen2.5:7b"
OLLAMA_URL      = "http://localhost:11434/api/generate"
MAX_FILINGS     = 40
SECTION_CHARS   = 6_000
FETCH_WORKERS   = 8
LLM_WORKERS     = 4
SEC_MAX_PER_SEC = 8

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
SEC_HEADERS = {"User-Agent": SEC_EMAIL}


class RateLimiter:
    def __init__(self, max_per_sec: float):
        self.min_interval = 1.0 / max_per_sec
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            sleep_for = self.min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self.last_call = time.monotonic()


sec_rate_limiter = RateLimiter(SEC_MAX_PER_SEC)


# ---------------------------------------------------------------------------
# Company name <-> CIK lookup
# ---------------------------------------------------------------------------

_TICKERS_CACHE = CACHE_DIR / "company_tickers.json"


def load_company_index() -> pd.DataFrame:
    """SEC's official ticker->CIK->name mapping, cached locally for 1 day."""
    if _TICKERS_CACHE.exists() and (time.time() - _TICKERS_CACHE.stat().st_mtime) < 86400:
        data = json.loads(_TICKERS_CACHE.read_text())
    else:
        r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        _TICKERS_CACHE.write_text(json.dumps(data))
    rows = [{"cik": str(v["cik_str"]), "ticker": v["ticker"], "name": v["title"]} for v in data.values()]
    return pd.DataFrame(rows)


def search_companies(query: str, index: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    q = query.strip().lower()
    if not q:
        return index.head(0)
    mask = index["name"].str.lower().str.contains(q, na=False) | index["ticker"].str.lower().eq(q)
    return index[mask].head(limit)


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def get_filings(cik: str, years: list[int] | None, form_types=("10-K", "10-Q")) -> list[dict]:
    sec_rate_limiter.wait()
    url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
    r = requests.get(url, headers=SEC_HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    company_name = data.get("name", cik)
    recent = data["filings"]["recent"]
    results = []
    for i, form in enumerate(recent["form"]):
        if form not in form_types:
            continue
        filing_date = recent["filingDate"][i]
        if years and int(filing_date[:4]) not in years:
            continue
        results.append({
            "cik": cik, "company_name": company_name,
            "accession": recent["accessionNumber"][i],
            "filing_date": filing_date, "form": form,
            "primary_document": recent["primaryDocument"][i],
        })

    if years and min(years) < 2020:
        for f in data["filings"].get("files", []):
            sec_rate_limiter.wait()
            rr = requests.get(f"https://data.sec.gov/submissions/{f['name']}", headers=SEC_HEADERS, timeout=30)
            if rr.status_code != 200:
                continue
            arch = rr.json()
            for i, form in enumerate(arch["form"]):
                if form not in form_types:
                    continue
                filing_date = arch["filingDate"][i]
                if int(filing_date[:4]) not in years:
                    continue
                results.append({
                    "cik": cik, "company_name": company_name,
                    "accession": arch["accessionNumber"][i],
                    "filing_date": filing_date, "form": form,
                    "primary_document": arch["primaryDocument"][i],
                })
    return results[:MAX_FILINGS], company_name


def get_document_url(cik: str, accession: str, primary_document: str) -> str:
    accession_clean = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_clean}/{primary_document}"


def fetch_filing_text(url: str) -> str:
    cache_key = re.sub(r"[^a-zA-Z0-9]", "_", url)[-120:]
    cache_file = CACHE_DIR / f"{cache_key}.txt"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    sec_rate_limiter.wait()
    r = requests.get(url, headers=SEC_HEADERS, timeout=60)
    if r.status_code != 200:
        return ""
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" "))
    cache_file.write_text(text, encoding="utf-8")
    return text


_LEGAL_PATTERNS = [r"item\s+1\.?\s+legal\s+proceedings", r"item\s+3\.?\s+legal\s+proceedings"]
_NEXT_SECTION_PATTERNS = [r"item\s+[0-9]+[a-z]?\.", r"part\s+[ivxlc]+\b"]
_CROSS_REF_PATTERN = re.compile(r"\bsee\s+(item|note|part)\b", re.IGNORECASE)
_CONTINGENCY_HEADER = re.compile(
    r"(commitments\s+and\s+contingenc|litigation\s+and\s+contingenc|"
    r"loss\s+contingenc)",
    re.IGNORECASE,
)


def extract_contingency_note(text: str, window_chars: int = 6000, min_content_chars: int = 400) -> str:
    """Some companies (e.g. Amazon) leave Item 1 Legal Proceedings as a bare cross-reference
    to a financial-statement footnote like 'Note 4 — Commitments and Contingencies'. This finds
    that footnote directly, wherever it lives in the document."""
    matches = list(_CONTINGENCY_HEADER.finditer(text))
    for m in matches:
        start = m.start()
        end = min(len(text), start + window_chars)
        section = text[start:end]
        if len(section) >= min_content_chars:
            return section
    return ""


_8K_LITIGATION_KEYWORDS = re.compile(
    r"(lawsuit|litigation|settlement|settled|consent decree|consent order|"
    r"class action|complaint|verdict|judgment|injunction|exclusion order|"
    r"civil penalty|monetary penalty|plea agreement|indictment|subpoena)",
    re.IGNORECASE,
)


def extract_8k_section(text: str, window_chars: int = SECTION_CHARS) -> str:
    """8-Ks have no standard 'Item 1 Legal Proceedings' section - they're event-driven and
    their structure varies a lot (numbered Items like 1.01/8.01, often followed by a press
    release exhibit). Most 8-Ks aren't about litigation at all (earnings, exec changes,
    M&A), so first check whether this one is even relevant before wasting an LLM call.
    If it is, just take a window around the first litigation-flavored keyword hit - 8-Ks
    are short enough that this usually captures the whole relevant announcement."""
    if not text:
        return ""
    lower = text.lower()
    if len(text) <= window_chars:
        return text if _8K_LITIGATION_KEYWORDS.search(lower) else ""
    m = _8K_LITIGATION_KEYWORDS.search(lower)
    if not m:
        return ""
    start = max(0, m.start() - 500)  # a little context before the first hit
    end = min(len(text), start + window_chars)
    return text[start:end]


def extract_legal_section(text: str, min_content_chars: int = 400) -> str:
    lower = text.lower()
    starts = []
    for pat in _LEGAL_PATTERNS:
        for m in re.finditer(pat, lower):
            starts.append(m.start())
    if not starts:
        for kw in ("legal proceedings", "legal matters", "contingencies"):
            pos = lower.find(kw, 5_000)
            if pos != -1:
                starts.append(pos)
    section = ""
    if starts:
        for start in sorted(starts, reverse=True):
            end = min(len(text), start + SECTION_CHARS)
            snippet = lower[start:end]
            trim_pos = None
            for pat in _NEXT_SECTION_PATTERNS:
                m = re.search(pat, snippet[200:])
                if m:
                    candidate = 200 + m.start()
                    trim_pos = candidate if (trim_pos is None or candidate < trim_pos) else trim_pos
            section_end = start + trim_pos if (trim_pos and trim_pos > 500) else end
            candidate_section = text[start:section_end]
            if len(candidate_section) >= min_content_chars:
                section = candidate_section
                break
        if not section:
            section = text[sorted(starts)[-1]: sorted(starts)[-1] + SECTION_CHARS]

    # If what we found is just a cross-reference stub ("See Item 1 of Part I, Note 4..."),
    # go find the actual contingencies footnote it's pointing to instead.
    is_cross_ref = bool(_CROSS_REF_PATTERN.search(section[:300])) if section else True
    if not section or (is_cross_ref and len(section) < 1500):
        footnote = extract_contingency_note(text)
        if footnote:
            section = footnote  # real content only - drop the cross-reference stub entirely

    return section


# ---------------------------------------------------------------------------
# Regex layer
# ---------------------------------------------------------------------------

_MONEY_PATTERN = re.compile(
    r"(?P<currency>[\$€£])\s?(?P<amount>[\d,]+(?:\.\d+)?)\s?(?P<unit>million|billion|thousand|M|B|K)?"
    r"|\b(?P<amount2>[\d,]+(?:\.\d+)?)\s?(?P<unit2>million|billion)\b",
    re.IGNORECASE,
)
_CASE_NAME_PATTERN = re.compile(r"([A-Z][A-Za-z0-9&.,\-\s]{2,60}?\s+v\.?s?\.?\s+[A-Z][A-Za-z0-9&.,\-\s]{2,60})")
_YEAR_PATTERN = re.compile(r"\b(19[9]\d|20[0-3]\d)\b")
CURRENCY_MAP = {"$": "USD", "€": "EUR", "£": "GBP"}


def money_matches(text: str) -> list[dict]:
    out = []
    for m in _MONEY_PATTERN.finditer(text):
        currency = CURRENCY_MAP.get(m.group("currency"), "USD")
        amount = m.group("amount") or m.group("amount2")
        unit = (m.group("unit") or m.group("unit2") or "").lower()
        if not amount:
            continue
        try:
            num = float(amount.replace(",", ""))
        except ValueError:
            continue
        if "billion" in unit or unit == "b":
            num *= 1000
        elif "thousand" in unit or unit == "k":
            num /= 1000
        elif num > 100_000:
            num /= 1_000_000
        out.append({"currency": currency, "value_millions": round(num, 2)})
    return out


def regex_preextract(section_text: str) -> dict:
    money = money_matches(section_text)
    case_names = list({m.strip() for m in _CASE_NAME_PATTERN.findall(section_text)})
    years = sorted({int(y) for y in _YEAR_PATTERN.findall(section_text)})
    return {"money": money[:15], "case_names": case_names[:10], "years": years}


def build_hint_block(entities: dict) -> str:
    if not any(entities.values()):
        return "No numeric or case-name entities were pre-detected."
    lines = []
    if entities["money"]:
        amounts = ", ".join(f"{m['currency']} {m['value_millions']}M" for m in entities["money"])
        lines.append(f"Amounts found (converted to millions, currency preserved): {amounts}")
    if entities["case_names"]:
        lines.append(f"Case name patterns found: {'; '.join(entities['case_names'])}")
    if entities["years"]:
        lines.append(f"Years mentioned: {', '.join(str(y) for y in entities['years'])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama extraction
# ---------------------------------------------------------------------------

EXAMPLE_BLOCK = """EXAMPLE (for calibration only, not part of the filing you are analyzing):

Text: "In September 2021, the District Court ruled in favor of Epic on its claim that the
Company's anti-steering provisions violate California's Unfair Competition Law, and issued
an injunction. In November 2023, the Company reached a settlement with Beta Corp, agreeing
to pay $12 million over three years starting in 2024 to resolve the patent dispute."

Correct extraction:
{
  "cases": [
    {
      "case_name": "Epic Games, Inc. v. Apple Inc.",
      "counterparty": "Epic Games, Inc.",
      "nature_of_claim": "violation of California's Unfair Competition Law (anti-steering provisions)",
      "filing_status": "judgment",
      "outcome": "District Court ruled in favor of Epic on the UCL claim and issued an injunction",
      "settlement_amount": null,
      "settlement_currency": null,
      "evidence": "District Court ruled in favor of Epic ... issued an injunction"
    },
    {
      "case_name": "Beta Corp Patent Dispute",
      "counterparty": "Beta Corp",
      "nature_of_claim": "patent dispute",
      "filing_status": "settled",
      "outcome": "Settlement reached, $12 million payable over three years",
      "settlement_amount": 12.0,
      "settlement_currency": "USD",
      "evidence": "reached a settlement with Beta Corp, agreeing to pay $12 million"
    }
  ],
  "payment_schedule": [
    {"case_name": "Beta Corp Patent Dispute", "year": 2024, "amount": 4.0, "currency": "USD", "payment_note": "first of three annual installments"},
    {"case_name": "Beta Corp Patent Dispute", "year": 2025, "amount": 4.0, "currency": "USD", "payment_note": null},
    {"case_name": "Beta Corp Patent Dispute", "year": 2026, "amount": 4.0, "currency": "USD", "payment_note": null}
  ]
}

Note: the ruling/injunction case is "judgment" (no settlement, no amount - a court decided it).
The Beta Corp case is "settled" (negotiated agreement, has a dollar amount and a schedule).
Do not confuse the two.
"""

PROMPT_TEMPLATE = """You are a financial-legal data extraction specialist reading SEC filings.

{example_block}

Now extract ALL litigation matters from the ACTUAL filing text below.

DETECTED ENTITIES (pre-scanned with regex - anchor your extraction to these, but only
include amounts/cases relevant to litigation; ignore irrelevant numbers):
{hint_block}

Return ONLY a JSON object - no explanation, no markdown, just raw JSON.

JSON format:
{{
  "cases": [
    {{
      "case_name": "string - REQUIRED, never null. If no formal case name exists, write a short descriptive label, never leave blank",
      "counterparty": "string or null - the opposing party/plaintiff/regulator, NEVER the filer itself",
      "nature_of_claim": "string - REQUIRED, never null",
      "filing_status": "pending or settled or judgment or dismissed or appealed or null. Use 'settled' ONLY for a negotiated settlement agreement (usually involves a payment or agreed terms). Use 'judgment' for a case resolved by a court/jury ruling or verdict after trial, even if one side substantially won - a judgment is NOT a settlement.",
      "outcome": "string or null",
      "loss_estimate_low": null,
      "loss_estimate_high": null,
      "settlement_amount": null,
      "settlement_currency": "USD or EUR or GBP, only if settlement_amount is set",
      "total_liability": null,
      "evidence": "verbatim quote (<=25 words) from the text supporting this case entry"
    }}
  ],
  "payment_schedule": [
    {{
      "case_name": "string - must match a case_name above",
      "year": 2024,
      "amount": null,
      "currency": "USD or EUR or GBP",
      "payment_note": "string or null"
    }}
  ]
}}

Rules:
- Include EVERY lawsuit, investigation, regulatory action, settlement, class action mentioned.
- Every case MUST have both case_name and nature_of_claim filled in. If you cannot determine
  either one, DO NOT include that entry - no partial/empty stub entries.
- "settled" means a negotiated agreement was reached (often with a payment or agreed terms
  going forward). A court ruling/verdict/judgment after trial - even a mostly favorable one -
  is "judgment", NOT "settled". Do not use "settled" just because a case is no longer pending.
- If the SAME case is mentioned more than once, merge it into ONE case entry.
- counterparty must be the actual opposing party, NEVER the filer itself.
- All monetary values must be in millions as plain numbers. ALWAYS set the matching currency
  field - do not assume USD if the text says euro or pound.
- If a settlement has installments, create one payment_schedule entry per year.
- evidence must be an exact quote copied from the text, not a paraphrase.
- Use null for any field not mentioned. If no litigation found, return {{"cases": [], "payment_schedule": []}}.

SEC FILING TEXT:
{section_text}

Return JSON only:"""


def call_ollama(section_text: str, model: str, hint_block: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(example_block=EXAMPLE_BLOCK, hint_block=hint_block,
                                     section_text=section_text[:SECTION_CHARS])
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False, "format": "json",
                  "temperature": 0, "options": {"num_predict": 2048, "num_ctx": 8192}},
            timeout=300,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return {"cases": [], "payment_schedule": [], "_error": "Cannot connect to Ollama - is it running? (ollama serve)"}
    except requests.exceptions.Timeout:
        return {"cases": [], "payment_schedule": [], "_error": "Ollama timed out on this filing"}
    return parse_llm_output(response.json().get("response", ""))


def parse_llm_output(raw_text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", raw_text).strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            data.setdefault("cases", [])
            data.setdefault("payment_schedule", [])
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            data.setdefault("cases", [])
            data.setdefault("payment_schedule", [])
            return data
        except json.JSONDecodeError:
            pass
    return {"cases": [], "payment_schedule": []}


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def is_junk_case(case: dict) -> bool:
    name = (case.get("case_name") or "").strip()
    nature = (case.get("nature_of_claim") or "").strip()
    return not name and not nature


def fix_counterparty(case: dict) -> dict:
    bad_values = {"company", "the company", "registrant", "the registrant", ""}
    cp = (case.get("counterparty") or "").strip().lower()
    if cp in bad_values:
        m = re.search(r"v\.?s?\.?\s+(.+)", case.get("case_name") or "", re.IGNORECASE)
        case["counterparty"] = m.group(1).strip(" .") if m else None
    return case


def backfill_case_name(case: dict) -> dict:
    if case.get("case_name"):
        return case
    cp, nature = case.get("counterparty"), case.get("nature_of_claim")
    if cp and nature:
        case["case_name"] = f"{nature[:50]} ({cp})"
    elif nature:
        case["case_name"] = nature[:60]
    return case


def check_grounding(case: dict, section_text: str) -> dict:
    """Flag (don't silently trust) any case whose evidence quote can't be found in the
    source text. Uses a tolerant partial match (looks for the longest ~8-word run of the
    quote, not just the first 60 chars) so minor whitespace/formatting differences from
    the LLM's copy don't cause false negatives on genuinely correct extractions."""
    quote = case.get("evidence")
    section_norm = re.sub(r"\s+", " ", section_text).lower()
    if not quote:
        case["grounded"] = False
        return case
    q_norm = re.sub(r"\s+", " ", str(quote)).lower().strip()
    if len(q_norm) <= 3:
        case["grounded"] = False
        return case
    if q_norm in section_norm:
        case["grounded"] = True
        return case
    # tolerant fallback: check if most consecutive 8-word windows of the quote appear
    words = q_norm.split()
    if len(words) < 4:
        case["grounded"] = q_norm in section_norm
        return case
    window = 8
    windows = [" ".join(words[i:i + window]) for i in range(0, max(1, len(words) - window + 1), window)]
    hits = sum(1 for w in windows if w in section_norm)
    case["grounded"] = hits >= max(1, len(windows) // 2)
    return case


_SETTLEMENT_SIGNALS = re.compile(
    r"(settlement agreement|reached a settlement|agreed to pay|will pay|settled for|"
    r"in exchange for|mutually agreed|consent decree|agreed to resolve)",
    re.IGNORECASE,
)
_JUDGMENT_SIGNALS = re.compile(
    r"(court (ruled|found|held)|jury (found|returned)|trial court|verdict|"
    r"granted summary judgment|issued an injunction|court of appeals affirmed|"
    r"panel affirmed|ruling in favor)",
    re.IGNORECASE,
)
_REGULATORY_SIGNALS = re.compile(
    r"(exclusion order|cease.and.desist|consent order|"
    r"administrative law judge|itc (ruling|order|determination)|"
    r"limited exclusion|import ban|prohibiting importation)",
    re.IGNORECASE,
)


def check_status_consistency(case: dict) -> dict:
    """Cross-checks the LLM's filing_status claim against its own outcome text.
    Catches the common failure mode of labeling a court judgment - or a regulatory/
    administrative ruling like an ITC exclusion order - as 'settled'."""
    status = (case.get("filing_status") or "").strip().lower()
    outcome = case.get("outcome") or ""
    case["status_flagged"] = False
    if status == "settled":
        if not outcome.strip():
            # No supporting outcome text at all - can't verify "settled" against anything,
            # so treat it as unverified rather than silently trusting the bare label.
            case["status_flagged"] = True
            return case
        has_settlement_signal = bool(_SETTLEMENT_SIGNALS.search(outcome))
        has_judgment_signal = bool(_JUDGMENT_SIGNALS.search(outcome)) or bool(_REGULATORY_SIGNALS.search(outcome))
        if has_judgment_signal and not has_settlement_signal:
            case["status_flagged"] = True
    return case


def backfill_payment_schedule(cases: list[dict], payments: list[dict]) -> list[dict]:
    """Guarantees every case with a real (non-zero) settlement_amount/total_liability has
    at least one matching payment_schedule row. A stated amount of exactly 0 is treated as
    a likely hallucination/placeholder, not a real figure - the LLM has no reason to know a
    settlement was literally $0, and this pattern shows up when it should have said null."""
    payments_by_case = {}
    for p in payments:
        key = (p.get("case_name") or "").strip().lower()
        payments_by_case.setdefault(key, []).append(p)

    cleaned_payments = [p for p in payments if p.get("amount") not in (None, 0)]

    for case in cases:
        amount = case.get("settlement_amount")
        if amount in (None, 0):
            amount = case.get("total_liability")
        if amount in (None, 0):
            continue
        key = (case.get("case_name") or "").strip().lower()
        existing = [p for p in payments_by_case.get(key, []) if p.get("amount") not in (None, 0)]
        if existing:
            continue
        filing_year = int(case.get("filing_date", "0000")[:4]) if case.get("filing_date") else None
        cleaned_payments.append({
            "cik": case.get("cik"), "company_name": case.get("company_name"),
            "filing_date": case.get("filing_date"), "form": case.get("form"),
            "case_name": case.get("case_name"),
            "year": filing_year,
            "amount": amount,
            "currency": case.get("settlement_currency") or "USD",
            "amount_raw": amount,
            "amount_flagged": False,
            "payment_note": "No installment schedule disclosed - full amount shown as lump sum (backfilled from case settlement_amount, not a separate LLM extraction)",
            "backfilled": True,
        })
    return cleaned_payments


_CONTINUATION_SUFFIX = re.compile(
    r"\s*[\(\[]?\s*(continued|cont'?d|contd|part\s*\d+|cont\.)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)


def normalize_case_key(name: str) -> str:
    """Strips trailing continuation markers like '(continued)' or 'Part 2' so a single
    settlement that the LLM split across multiple entries (same case, different amount
    tranches) collapses to one dedup key instead of silently becoming separate 'cases'."""
    name = (name or "").strip().lower()
    prev = None
    while prev != name:
        prev = name
        name = _CONTINUATION_SUFFIX.sub("", name).strip()
    return name


def dedup_cases(cases: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for case in cases:
        key = normalize_case_key(case.get("case_name")) or f"__unnamed_{id(case)}"
        if key not in merged:
            merged[key] = dict(case)
            continue
        existing = merged[key]
        for field, value in case.items():
            if value in (None, ""):
                continue
            existing_val = existing.get(field)
            if existing_val in (None, ""):
                existing[field] = value
            elif field in ("filing_status", "outcome") and str(value) != str(existing_val):
                existing[field] = f"{existing_val}; {value}"
            elif field in ("loss_estimate_low", "loss_estimate_high", "total_liability"):
                try:
                    if float(value) > float(existing_val):
                        existing[field] = value
                except (TypeError, ValueError):
                    pass
            elif field == "settlement_amount":
                # Don't take the max here - a "(continued)" entry is usually a separate,
                # often contingent, additional tranche (e.g. "$125M more if the fund runs
                # out"), not a bigger/better version of the same number. Sum them instead,
                # and record the pieces so the total is auditable rather than silently
                # replacing one real figure with another.
                try:
                    combined = float(existing_val) + float(value)
                    existing.setdefault("settlement_amount_components", []).append(
                        f"{existing_val}+{value}={combined}"
                    )
                    existing[field] = combined
                except (TypeError, ValueError):
                    pass
    return list(merged.values())


def validate_amounts(case: dict, regex_money: list[dict]) -> dict:
    """Strict mode: a numeric field is only kept if it's independently confirmed by a
    regex-extracted dollar figure in the same currency, present in the source text. If not
    confirmed, the value is NULLED (not just flagged) - an unverifiable number is worse than
    no number, since it looks authoritative but may be hallucinated."""
    fields = ["loss_estimate_low", "loss_estimate_high", "settlement_amount", "total_liability"]
    case_currency = case.get("settlement_currency") or "USD"
    same_currency_amounts = [m["value_millions"] for m in regex_money if m["currency"] == case_currency]
    flagged = False
    for field in fields:
        val = case.get(field)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            case[f"{field}_unverified_raw"] = case.get(field)  # keep the un-parseable raw value visible
            case[field] = None
            flagged = True
            continue
        if val == 0:
            case[field] = None
            case[f"{field}_unverified_raw"] = 0  # keep a trace of the discarded placeholder value
            flagged = True
            continue
        verified = same_currency_amounts and any(abs(val - r) <= max(0.02 * r, 0.5) for r in same_currency_amounts)
        if not verified:
            case[field] = None
            case[f"{field}_unverified_raw"] = val  # keep the discarded value visible for manual review
            flagged = True
    case["amount_flagged"] = flagged
    # currency is only meaningful alongside an amount - drop an orphaned currency left over
    # from a nulled settlement_amount/total_liability so it doesn't look like there's a
    # figure attached when there isn't.
    if case.get("settlement_amount") is None and case.get("total_liability") is None:
        case["settlement_currency"] = None
    return case


_CASE_NAME_STOPWORDS = {
    "the", "and", "inc", "inc.", "corp", "corporation", "llc", "co", "company",
    "litigation", "lawsuit", "action", "actions", "class", "case", "matter",
    "mdl", "multidistrict", "v", "vs", "v.", "u.s.", "us", "of", "in", "re",
}


def _case_name_tokens(name: str) -> set:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return {w for w in words if w not in _CASE_NAME_STOPWORDS and len(w) > 2}


def _group_cases_fuzzy(cases: list[dict]) -> list[list[dict]]:
    """Groups cases across filings that are almost certainly the same underlying matter,
    even when the LLM phrased the name slightly differently each quarter (e.g.
    'Multidistrict Litigation - U.S. Consumer Class Actions' vs 'U.S. Consumer MDL
    Litigation'). Exact-string matching misses this; a >=40% token overlap on the
    meaningful words (ignoring generic legal boilerplate words) catches it instead."""
    n = len(cases)
    token_sets = [_case_name_tokens(c.get("case_name")) for c in cases]
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        if not token_sets[i]:
            continue
        for j in range(i + 1, n):
            if not token_sets[j]:
                continue
            overlap = token_sets[i] & token_sets[j]
            union_size = token_sets[i] | token_sets[j]
            if union_size and len(overlap) / len(union_size) >= 0.4:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, c in enumerate(cases):
        groups.setdefault(find(i), []).append(c)
    return list(groups.values())


def check_settlement_timeline(cases: list[dict]) -> list[dict]:
    """Cross-filing sanity check on likely-the-same case across quarters (matched fuzzily,
    since the LLM often renames the same matter slightly each filing). Flags two patterns:
    1. 'settled' reverting to 'pending' in a later filing - real settlements don't un-settle.
    2. An early 'settled' claim whose amount was unverifiable/hallucinated (amount_flagged),
       later corroborated by the same matter reporting a clean, verified settlement amount -
       a strong sign the early entry was premature or fabricated, since amounts don't
       reliably firm up like that unless the first one was wrong.
    """
    for group in _group_cases_fuzzy(cases):
        if len(group) < 2:
            for c in group:
                c.setdefault("timeline_flagged", False)
            continue
        group.sort(key=lambda c: c.get("filing_date") or "")
        for c in group:
            c.setdefault("timeline_flagged", False)
        for i, earlier in enumerate(group):
            earlier_status = (earlier.get("filing_status") or "").strip().lower()
            if earlier_status != "settled":
                continue
            for later in group[i + 1:]:
                later_status = (later.get("filing_status") or "").strip().lower()
                if later_status == "pending":
                    earlier["timeline_flagged"] = True
                    earlier["status_flagged"] = True
                    later["timeline_flagged"] = True
                elif (later_status == "settled" and earlier.get("amount_flagged")
                        and not later.get("amount_flagged")
                        and later.get("settlement_amount") is not None):
                    earlier["timeline_flagged"] = True
                    earlier["status_flagged"] = True
    return cases


def normalize_payment_amount(pmt: dict) -> dict:
    amount = pmt.get("amount")
    pmt["amount_flagged"] = False
    pmt["amount_raw"] = amount
    pmt.setdefault("currency", "USD")
    if amount is None:
        return pmt
    try:
        val = float(amount)
    except (TypeError, ValueError):
        return pmt
    if val > 100_000:
        pmt["amount"] = round(val / 1_000_000, 2)
        pmt["amount_flagged"] = True
    return pmt


def confidence_score(case: dict) -> int:
    score = 0
    if case.get("case_name"): score += 25
    if case.get("counterparty"): score += 15
    if case.get("nature_of_claim"): score += 15
    if case.get("filing_status"): score += 10
    if case.get("settlement_amount") is not None: score += 20
    elif case.get("total_liability") is not None: score += 15
    elif case.get("loss_estimate_low") is not None: score += 10
    if case.get("grounded"): score += 15
    else: score -= 15
    if case.get("amount_flagged"): score -= 20
    if case.get("status_flagged"): score -= 25
    if case.get("timeline_flagged"): score -= 20
    return max(0, min(score, 100))


# ---------------------------------------------------------------------------
# Orchestration for the UI: fetch -> extract -> return dataframes for THIS run only
# ---------------------------------------------------------------------------

def run_extraction_for_filing(filing: dict, model: str) -> dict:
    url = get_document_url(filing["cik"], filing["accession"], filing["primary_document"])
    text = fetch_filing_text(url)
    if filing["form"] == "8-K":
        section = extract_8k_section(text) if text else ""
    else:
        section = extract_legal_section(text) if text else ""
    if not section:
        return {"cases": [], "payments": []}

    entities = regex_preextract(section)
    hint_block = build_hint_block(entities)
    parsed = call_ollama(section, model, hint_block)

    cases = [c for c in parsed.get("cases", []) if not is_junk_case(c)]
    cases = [backfill_case_name(c) for c in cases]
    cases = [fix_counterparty(c) for c in cases]
    cases = [check_grounding(c, section) for c in cases]
    cases = [check_status_consistency(c) for c in cases]
    cases = dedup_cases(cases)
    cases = [validate_amounts(c, entities["money"]) for c in cases]

    meta = {"cik": filing["cik"], "company_name": filing["company_name"],
            "filing_date": filing["filing_date"], "form": filing["form"]}
    for c in cases:
        c["confidence_score"] = confidence_score(c)
        c.update(meta)

    payments = [normalize_payment_amount(p) for p in parsed.get("payment_schedule", [])]
    for p in payments:
        p.update(meta)
        p.setdefault("backfilled", False)
    payments = backfill_payment_schedule(cases, payments)

    return {"cases": cases, "payments": payments}


def run_for_ui(cik: str, years: list[int], model: str = OLLAMA_MODEL,
                fetch_workers: int = FETCH_WORKERS, llm_workers: int = LLM_WORKERS,
                include_8k: bool = False,
                progress_callback=None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Runs the full pipeline for one CIK + year list. Returns (cases_df, payments_df,
    company_name) scoped to THIS run only (not the accumulated historical CSV).
    include_8k=True also scans 8-K filings, which often carry a settlement's exact dollar
    figure days after signing - well before it shows up summarized in the next 10-Q/10-K."""
    form_types = ("10-K", "10-Q", "8-K") if include_8k else ("10-K", "10-Q")
    filings, company_name = get_filings(cik, years, form_types=form_types)
    if not filings:
        return pd.DataFrame(), pd.DataFrame(), company_name

    if progress_callback:
        progress_callback(0, len(filings), "Fetching filings...")

    all_cases, all_payments = [], []
    done = 0
    with ThreadPoolExecutor(max_workers=llm_workers) as executor:
        futures = {executor.submit(run_extraction_for_filing, f, model): f for f in filings}
        for future in as_completed(futures):
            done += 1
            try:
                result = future.result()
                all_cases.extend(result["cases"])
                all_payments.extend(result["payments"])
            except Exception:
                pass
            if progress_callback:
                progress_callback(done, len(filings), f"Processed {done}/{len(filings)} filings")

    cases_df = pd.DataFrame(all_cases)
    payments_df = pd.DataFrame(all_payments)

    if not cases_df.empty:
        all_cases = check_settlement_timeline(all_cases)
        for c in all_cases:
            c["confidence_score"] = confidence_score(c)
        cases_df = pd.DataFrame(all_cases)

    CASE_COLS = ["cik", "company_name", "filing_date", "form", "case_name", "counterparty",
                 "nature_of_claim", "filing_status", "status_flagged", "timeline_flagged", "outcome",
                 "loss_estimate_low", "loss_estimate_high", "settlement_amount",
                 "settlement_amount_unverified_raw", "settlement_currency", "total_liability",
                 "total_liability_unverified_raw", "confidence_score", "grounded", "amount_flagged"]
    PMT_COLS = ["cik", "company_name", "filing_date", "form", "case_name", "year",
                "amount", "currency", "amount_raw", "amount_flagged", "backfilled", "payment_note"]

    if not cases_df.empty:
        for c in CASE_COLS:
            if c not in cases_df.columns:
                cases_df[c] = None
        cases_df = cases_df[CASE_COLS].sort_values("filing_date")
    if not payments_df.empty:
        for c in PMT_COLS:
            if c not in payments_df.columns:
                payments_df[c] = None
        payments_df = payments_df[PMT_COLS].sort_values(["case_name", "year"])

    return cases_df, payments_df, company_name