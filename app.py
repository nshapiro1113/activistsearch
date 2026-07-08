"""
Streamlit web UI for the activist-candidate screener.

Run with:
    streamlit run app.py

Flow: enter tickers -> score them (Factors 1 & 3 computed from financials,
Factors 2/4/5 drafted by Claude) -> check the boxes for candidates worth a
closer look -> generate a full playbook/report for just those.
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from scorer import CapitalIQClient, MockCapitalIQClient, score_company
from playbook import generate_playbook
from run_screen import build_peer_groups

st.set_page_config(page_title="Activist Candidate Screener", layout="wide")

if "financials_by_ticker" not in st.session_state:
    st.session_state.financials_by_ticker = {}
if "peer_groups" not in st.session_state:
    st.session_state.peer_groups = {}
if "scores" not in st.session_state:
    st.session_state.scores = {}
if "playbooks" not in st.session_state:
    st.session_state.playbooks = {}

st.title("Activist Investment Candidate Screener")
st.caption("Score candidates against the 5-factor methodology, then generate a full playbook for the ones worth a closer look.")

with st.sidebar:
    st.header("Data source")
    use_mock = st.toggle(
        "Use mock financial data",
        value=not bool(os.environ.get("CIQ_USERNAME")),
        help="No Capital IQ credentials needed. Turn this off once you have real CIQ credentials.",
    )

    ciq_username = ciq_password = None
    if not use_mock:
        ciq_username = st.text_input("CIQ username", value=os.environ.get("CIQ_USERNAME", ""))
        ciq_password = st.text_input("CIQ password", value=os.environ.get("CIQ_PASSWORD", ""), type="password")

    st.header("Claude API")
    anthropic_key = st.text_input(
        "Anthropic API key",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Needed for Factor 2/4/5 scoring and for playbook generation. Get one at console.anthropic.com.",
    )

    st.divider()
    if st.button("Reset session", help="Clear all scored candidates and generated playbooks."):
        st.session_state.financials_by_ticker = {}
        st.session_state.peer_groups = {}
        st.session_state.scores = {}
        st.session_state.playbooks = {}
        st.rerun()


def get_anthropic_client():
    if not anthropic_key:
        st.error("Enter an Anthropic API key in the sidebar first.")
        st.stop()
    import anthropic
    return anthropic.Anthropic(api_key=anthropic_key)


def get_ciq_client():
    if use_mock:
        return MockCapitalIQClient()
    if not ciq_username or not ciq_password:
        st.error("Enter your Capital IQ username and password in the sidebar, or turn on mock data.")
        st.stop()
    return CapitalIQClient(username=ciq_username, password=ciq_password)


st.subheader("1. Enter tickers to screen")
tickers_input = st.text_area(
    "Tickers (comma, space, or newline separated)",
    placeholder="CP, CNI, NSC, UNP",
    height=80,
)
shared_notes = st.text_area(
    "Optional research notes (shared context for all tickers below, e.g. recent news, CEO tenure, ownership structure)",
    placeholder="e.g. NSC had a derailment/CEO scrutiny in 2023-2024; CP completed the KCS merger in 2023...",
    height=80,
)

if st.button("Score candidates", type="primary"):
    raw = tickers_input.replace(",", "\n").replace(" ", "\n")
    tickers = list(dict.fromkeys(t.strip().upper() for t in raw.splitlines() if t.strip()))
    if not tickers:
        st.warning("Enter at least one ticker.")
    else:
        ciq_client = get_ciq_client()
        anthropic_client = get_anthropic_client()

        financials_by_ticker = {}
        progress = st.progress(0.0, text="Fetching fundamentals...")
        for i, ticker in enumerate(tickers):
            try:
                financials_by_ticker[ticker] = ciq_client.get_fundamentals(ticker)
            except Exception as exc:
                st.warning(f"Failed to fetch {ticker}: {exc}")
            progress.progress((i + 1) / len(tickers), text=f"Fetched {ticker}")

        if not financials_by_ticker:
            st.error("Could not fetch fundamentals for any ticker.")
            st.stop()

        peer_groups = build_peer_groups(financials_by_ticker)

        scores = {}
        progress = st.progress(0.0, text="Scoring candidates...")
        for i, (ticker, company) in enumerate(financials_by_ticker.items()):
            try:
                scores[ticker] = score_company(
                    company, peer_groups[ticker],
                    research_notes=shared_notes,
                    anthropic_client=anthropic_client,
                )
            except Exception as exc:
                st.warning(f"Failed to score {ticker}: {exc}")
            progress.progress((i + 1) / len(financials_by_ticker), text=f"Scored {ticker}")

        st.session_state.financials_by_ticker = financials_by_ticker
        st.session_state.peer_groups = peer_groups
        st.session_state.scores = scores
        st.session_state.playbooks = {}
        st.success(f"Scored {len(scores)} of {len(tickers)} ticker(s).")


if st.session_state.scores:
    st.subheader("2. Results — pick candidates to investigate further")

    ranked = sorted(st.session_state.scores.values(), key=lambda s: s.total, reverse=True)
    df = pd.DataFrame([s.as_row() for s in ranked])
    df.insert(0, "Investigate?", False)
    # Keep the score board compact; the thesis gets its own full-width column.
    score_cols = ["ticker", "total", "band", "factor1_gap", "factor2_fix",
                  "factor3_balance_sheet", "factor4_catalyst", "factor5_feasibility"]
    df = df[["Investigate?"] + score_cols + ["thesis"]]

    edited = st.data_editor(
        df,
        column_config={
            "Investigate?": st.column_config.CheckboxColumn(required=True),
            "total": st.column_config.NumberColumn("Total", format="%.1f"),
            "thesis": st.column_config.TextColumn("One-sentence thesis", width="large"),
        },
        disabled=[c for c in df.columns if c != "Investigate?"],
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Thesis, one sentence per opportunity:**")
    for s in ranked:
        st.markdown(f"- **[{s.total:.1f}] {s.ticker}** — {s.thesis}")

    with st.expander("Factor rationale (why each score is what it is)"):
        for s in ranked:
            st.markdown(f"**{s.ticker} — {s.name}**")
            for label, factor in [
                ("Factor 1 (Performance Gap)", s.factor1),
                ("Factor 2 (Credible Fix)", s.factor2),
                ("Factor 3 (Balance Sheet Slack)", s.factor3),
                ("Factor 4 (Catalyst/Governance)", s.factor4),
                ("Factor 5 (Structural Feasibility)", s.factor5),
            ]:
                st.markdown(f"- *{label}* ({factor.score:.1f}/{factor.max_score:.0f}): {factor.rationale}")
            notes = s.qualitative_notes
            if notes.get("named_operator_or_asset"):
                st.markdown(f"- *Named operator/asset:* {notes['named_operator_or_asset']}")
            if notes.get("red_flags"):
                st.markdown(f"- *Red flags:* {', '.join(notes['red_flags'])}")
            st.divider()

    selected_tickers = edited.loc[edited["Investigate?"], "ticker"].tolist()

    if st.button("Generate playbooks for selected", disabled=not selected_tickers):
        anthropic_client = get_anthropic_client()
        for ticker in selected_tickers:
            if ticker in st.session_state.playbooks:
                continue
            with st.spinner(f"Writing playbook for {ticker}..."):
                try:
                    text = generate_playbook(
                        st.session_state.scores[ticker],
                        st.session_state.financials_by_ticker[ticker],
                        st.session_state.peer_groups[ticker],
                        research_notes=shared_notes,
                        anthropic_client=anthropic_client,
                    )
                    st.session_state.playbooks[ticker] = text
                except Exception as exc:
                    st.warning(f"Failed to generate playbook for {ticker}: {exc}")

if st.session_state.playbooks:
    st.subheader("3. Playbooks / full reports")
    for ticker, text in st.session_state.playbooks.items():
        with st.expander(f"{ticker} playbook", expanded=True):
            st.markdown(text)
            st.download_button(
                "Download as markdown",
                data=text,
                file_name=f"{ticker}_playbook.md",
                mime="text/markdown",
                key=f"download_{ticker}",
            )
