# SEC Litigation Extractor

A local, LLM-assisted pipeline that scans a company's SEC 10-K/10-Q (and optionally 8-K) filings for litigation disclosures, extracts structured case data and payment schedules, and surfaces it through a Streamlit UI.

---

## 1. What it does

Given a company (by name/ticker or CIK) and one or more years, the pipeline:

1. Looks up the company's CIK and pulls its filing list from SEC EDGAR.
2. Fetches each relevant 10-K/10-Q (and optionally 8-K) document.
3. Locates the "Legal Proceedings" section (or the linked contingencies footnote, or a keyword-matched window for 8-Ks).
4. Pre-scans that section with regex to detect dollar amounts, candidate case names, and years — these become "hints" for the LLM.
5. Sends the section + hints to a local Ollama model (`qwen2.5:7b` by default) with a calibration example, asking for structured JSON: a list of litigation **cases** and a **payment schedule**.
6. Runs the extraction through a series of accuracy/validation passes (below).
7. Returns two pandas DataFrames — cases and payments — scoped to the current run, and lets the user download them as CSV.

## 2. Files

| File | Purpose |
|---|---|
| `pipeline_core.py` | All extraction logic: SEC fetching, section-finding, regex pre-extraction, Ollama calls, post-processing/validation, and the `run_for_ui` orchestrator. |
| `app.py` | Streamlit front end. Company search, year/model selection, progress bar, results tables, CSV downloads. |

## 3. Setup

```bash
pip install streamlit pandas requests beautifulsoup4 --break-system-packages
ollama pull qwen2.5:7b
OLLAMA_NUM_PARALLEL=4 ollama serve      # keep running in a separate terminal
```

Edit `SEC_EMAIL` in `pipeline_core.py` to your real email address — SEC EDGAR requires a contact email in the User-Agent header for all requests.

## 4. Running it

```bash
streamlit run app.py
```

In the UI:
- Search for a company by name/ticker, or paste a CIK directly.
- Pick one or more years.
- Optionally enable **8-K scanning** — 8-Ks are event-driven and often carry the exact settlement dollar figure days after signing, before it appears summarized in the next 10-Q/10-K. This makes the run slower since most 8-Ks aren't litigation-related.
- Click **Run extraction**. Results appear as two tables (Cases, Payment Schedule) with CSV download buttons.

## 5. Pipeline internals

### 5.1 Filing retrieval
- `get_filings()` pulls the submissions JSON for a CIK, filters to the requested form types and years, and paginates into older archived filing indexes when a requested year predates 2020 (EDGAR's "recent" feed doesn't go back that far).
- Capped at `MAX_FILINGS = 40` filings per run.
- A `RateLimiter` enforces `SEC_MAX_PER_SEC = 8` requests/second across all threads to stay within SEC's fair-access guidelines.
- Fetched filing text is cached to disk (`.cache/`) so re-runs don't re-download.

### 5.2 Section extraction
- `extract_legal_section()` looks for "Item 1/3 Legal Proceedings" headers, trims at the next Item/Part boundary, and falls back to keyword search if no header matches.
- If what's found is just a cross-reference stub (e.g. "See Note 4"), `extract_contingency_note()` locates the actual "Commitments and Contingencies" footnote instead and uses that.
- `extract_8k_section()` handles 8-Ks separately: no standard section headers exist, so it keyword-matches litigation-flavored terms and takes a window around the first hit, skipping non-litigation 8-Ks entirely before spending an LLM call.

### 5.3 Regex pre-extraction
- `money_matches()` finds dollar/euro/pound figures and normalizes them to millions.
- `regex_preextract()` also grabs candidate "X v. Y" case names and years mentioned, and packages all of it into a hint block that's injected into the LLM prompt so the model has concrete numbers/names to anchor to rather than free-associating.

### 5.4 LLM extraction
- One prompt per filing section, sent to Ollama's `/api/generate` endpoint.
- The prompt includes a worked example distinguishing "judgment" (court ruling) from "settled" (negotiated agreement) — a distinction the model tends to blur — and requires a short verbatim evidence quote per case for later grounding checks.
- Runs concurrently across filings via a `ThreadPoolExecutor` (`LLM_WORKERS = 4` by default).

### 5.5 Post-processing / validation (the accuracy layer)
Applied per case, after parsing:

- **`is_junk_case`** — drops boilerplate/non-cases.
- **`backfill_case_name`** / **`fix_counterparty`** — fills missing names, ensures the counterparty is never the filer itself.
- **`check_grounding`** — confirms the LLM's evidence quote actually appears in the source section; flags ungrounded entries.
- **`check_status_consistency`** — flags a "settled" status that has no supporting outcome text, or whose outcome text actually reads like a judgment/regulatory action rather than a negotiated settlement.
- **`dedup_cases`** — merges "(continued)"-style split entries via `normalize_case_key()`; sums settlement amounts across split tranches (rather than taking the max) and records the components for auditability.
- **`validate_amounts`** — strict mode: a numeric field is only kept if independently confirmed by a regex-detected dollar figure in the same currency and within ~2% tolerance. Unconfirmed values are **nulled**, not just flagged, since an unverifiable number looks authoritative but may be hallucinated; the raw value is preserved in a `_unverified_raw` field for manual review.
- **`normalize_payment_amount`** — flags and rescales amounts that look like they're in raw dollars rather than millions (`> 100,000`).
- **`backfill_payment_schedule`** — guarantees every case with a real settlement/liability amount has at least one payment-schedule row, even if the LLM didn't produce an explicit schedule (treats a literal `$0` as a likely placeholder, not a real figure).
- **`check_settlement_timeline`** (cross-filing) — fuzzy-groups the same underlying matter across quarters by token overlap on case names, then flags: (1) a "settled" status reverting to "pending" later — real settlements don't un-settle — and (2) an early flagged/unverified settlement amount later corroborated by a clean verified figure for the same matter — a sign the early entry was likely premature or fabricated.
- **`confidence_score`** — a 0–100 composite score per case, rewarding completeness and grounding, penalizing flags.

### 5.6 Orchestration
`run_for_ui()` ties it all together: fetch filings → run extraction concurrently per filing → run the cross-filing timeline check once all results are in → assemble two DataFrames with fixed column ordering, sorted by filing date (cases) or case name/year (payments).

## 6. Output columns

**Cases:** `cik, company_name, filing_date, form, case_name, counterparty, nature_of_claim, filing_status, status_flagged, timeline_flagged, outcome, loss_estimate_low, loss_estimate_high, settlement_amount, settlement_amount_unverified_raw, settlement_currency, total_liability, total_liability_unverified_raw, confidence_score, grounded, amount_flagged`

**Payments:** `cik, company_name, filing_date, form, case_name, year, amount, currency, amount_raw, amount_flagged, backfilled, payment_note`

## 7. Known limitations

- LLM-based extraction can still hallucinate or miss nuance despite the validation layer; `confidence_score`, `grounded`, and the various `*_flagged` columns are there to prioritize manual review, not replace it.
- Regex-based section/money detection is heuristic and can miss non-standard filing formats.
- Runs entirely locally against Ollama — extraction quality is bounded by the chosen model (`qwen2.5:7b` by default).
- Rate-limited to SEC's guidelines, so large multi-year, multi-company runs will take time.
