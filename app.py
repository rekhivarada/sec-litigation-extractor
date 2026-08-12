"""
Streamlit UI for the SEC litigation extraction pipeline.

SETUP (once):
    pip install streamlit pandas requests beautifulsoup4 --break-system-packages
    ollama pull qwen2.5:7b
    OLLAMA_NUM_PARALLEL=4 ollama serve      <- keep running in a separate terminal

RUN:
    streamlit run app.py

Put pipeline_core.py in the SAME folder as this file.
"""

import re
from datetime import datetime

import streamlit as st
import pandas as pd

import pipeline_core as core

st.set_page_config(page_title="Legal Losses Extractor", layout="wide")


def safe_filename_part(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("SEC Legal Losses Extractor")
st.caption("Select a company and year(s) — scans 10-K/10-Q filings for litigation disclosures, "
           "progress, outcomes, and payment schedules.")

if "company_index" not in st.session_state:
    with st.spinner("Loading SEC company list..."):
        st.session_state.company_index = core.load_company_index()

col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input("Company name or ticker", placeholder="e.g. Apple, AAPL, Johnson & Johnson")
    selected_cik, selected_name = None, None
    if query:
        matches = core.search_companies(query, st.session_state.company_index)
        if not matches.empty:
            options = {f"{r['name']} ({r['ticker']}) — CIK {r['cik']}": (r['cik'], r['name'])
                       for _, r in matches.iterrows()}
            picked = st.selectbox("Matches", list(options.keys()))
            selected_cik, selected_name = options[picked]
        else:
            st.warning("No matches. Try a different name/ticker, or enter CIK directly below.")

    manual_cik = st.text_input("...or enter CIK directly", value=selected_cik or "")

with col2:
    current_year = datetime.now().year
    years = st.multiselect("Year(s)", list(range(current_year, 2000, -1)),
                            default=[current_year - 1])
    model = st.text_input("Ollama model", value="qwen2.5:7b")
    include_8k = st.checkbox(
        "Also scan 8-K filings",
        value=False,
        help="8-Ks are event-driven and filed within 4 business days of a material event "
             "(like a settlement). They often carry the exact dollar figure first, before "
             "it's summarized in the next 10-Q/10-K - but there are many more of them per "
             "year and most aren't litigation-related, so this run takes longer.",
    )
    run_btn = st.button("Run extraction", type="primary", use_container_width=True)

st.divider()

if run_btn:
    cik = (manual_cik or selected_cik or "").strip()
    if not cik:
        st.error("Enter a CIK or select a company.")
        st.stop()
    if not years:
        st.error("Select at least one year.")
        st.stop()

    progress_bar = st.progress(0, text="Starting...")

    def on_progress(done, total, msg):
        progress_bar.progress(done / total if total else 0, text=msg)

    with st.spinner("Fetching filings and running extraction (this can take a few minutes)..."):
        try:
            cases_df, payments_df, company_name = core.run_for_ui(
                cik, years, model=model, include_8k=include_8k, progress_callback=on_progress
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    progress_bar.empty()
    st.session_state.last_result = (cases_df, payments_df, company_name, cik, years)

if "last_result" in st.session_state:
    cases_df, payments_df, company_name, cik, years = st.session_state.last_result

    st.subheader(f"{company_name} — CIK {cik} — {', '.join(str(y) for y in sorted(years))}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Cases found", len(cases_df))
    c2.metric("Payment schedule rows", len(payments_df))
    if not cases_df.empty and "grounded" in cases_df.columns:
        c3.metric("Ungrounded (needs review)", int((~cases_df["grounded"].astype(bool)).sum()))

    st.markdown("**Litigation Cases**")
    if cases_df.empty:
        st.info("No litigation disclosed for this company/year.")
    else:
        st.dataframe(cases_df, use_container_width=True)

    st.markdown("**Annual Payment Schedule**")
    if payments_df.empty:
        st.info("No multi-year payment schedule disclosed for this company/year.")
    else:
        st.dataframe(payments_df, use_container_width=True)

    year_str = "_".join(str(y) for y in sorted(years))
    base_name = f"{safe_filename_part(company_name)}_{year_str}"

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Litigation Cases CSV",
            data=cases_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_name}_cases.csv",
            mime="text/csv",
            type="primary",
            disabled=cases_df.empty,
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download Payment Schedule CSV",
            data=payments_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{base_name}_payments.csv",
            mime="text/csv",
            disabled=payments_df.empty,
            use_container_width=True,
        )