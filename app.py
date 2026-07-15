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
import statistics

import pandas as pd
import streamlit as st

from scorer import CapitalIQClient, FreeDataClient, MockCapitalIQClient, score_company
from playbook import generate_playbook
from run_screen import augment_peers_via_web_search, build_peer_groups

st.set_page_config(page_title="Activist Candidate Screener", layout="wide")

if "financials_by_ticker" not in st.session_state:
    st.session_state.financials_by_ticker = {}
if "peer_groups" not in st.session_state:
    st.session_state.peer_groups = {}
if "scores" not in st.session_state:
    st.session_state.scores = {}
if "playbooks" not in st.session_state:
    st.session_state.playbooks = {}

st.title(":green[activistsearch]")
st.caption(":green[Ned Shapiro]")
st.caption("Score candidates against the 5-factor methodology, then generate a full playbook for the ones worth a closer look.")

with st.sidebar:
    st.header("Data source")
    data_source = st.segmented_control(
        "Financial data source",
        ["Free (SEC + Yahoo)", "Capital IQ", "Mock"],
        default="Capital IQ" if os.environ.get("CIQ_USERNAME") else "Free (SEC + Yahoo)",
        label_visibility="collapsed",
    )

    ciq_username = ciq_password = None
    if data_source == "Capital IQ":
        ciq_username = st.text_input("CIQ username", value=os.environ.get("CIQ_USERNAME", ""))
        ciq_password = st.text_input("CIQ password", value=os.environ.get("CIQ_PASSWORD", ""), type="password")
    elif data_source == "Free (SEC + Yahoo)":
        st.caption("SEC EDGAR (XBRL filings) + Yahoo Finance. No credentials needed, but only covers "
                   "US SEC filers, derives EBITDA rather than using a vendor figure, and groups peers "
                   "by SIC code instead of GICS sub-industry.")

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
    if data_source == "Mock":
        return MockCapitalIQClient()
    if data_source == "Free (SEC + Yahoo)":
        return FreeDataClient()
    if not ciq_username or not ciq_password:
        st.error("Enter your Capital IQ username and password in the sidebar, or switch data source.")
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

        with st.spinner("Checking peer coverage (searching the web for direct competitors where needed)..."):
            augment_peers_via_web_search(financials_by_ticker, peer_groups, ciq_client, anthropic_client)

        scores = {}
        progress = st.progress(0.0, text="Scoring candidates...")
        for i, ticker in enumerate(tickers):
            if ticker not in financials_by_ticker:
                continue
            company = financials_by_ticker[ticker]
            try:
                scores[ticker] = score_company(
                    company, peer_groups[ticker],
                    research_notes=shared_notes,
                    anthropic_client=anthropic_client,
                )
            except Exception as exc:
                st.warning(f"Failed to score {ticker}: {exc}")
            progress.progress((i + 1) / len(tickers), text=f"Scored {ticker}")

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

    def _peer_median(peers, attr):
        vals = [getattr(p, attr) for p in peers if getattr(p, attr) is not None]
        return statistics.median(vals) if vals else None

    def _pct(value):
        return f"{value * 100:.1f}%" if value is not None else "n/a"

    show_overview_and_management = st.toggle(
        "Show company overview & management assessment", value=True,
        help="Turn off to keep each candidate's expander compact -- factor scores, peer metrics, "
             "and growth potential stay visible either way.",
    )

    st.markdown("### Opportunity detail (for each candidate)")
    for s in ranked:
        company = st.session_state.financials_by_ticker.get(s.ticker)
        peers = st.session_state.peer_groups.get(s.ticker, [])
        band_label, posture = s.band

        with st.expander(f"{s.ticker} — {s.name}  ·  {s.total:.1f}/100  ·  {band_label}"):
            header_cols = st.columns([1, 5])
            with header_cols[0]:
                if s.logo_url:
                    st.image(s.logo_url, width=64)
            with header_cols[1]:
                st.markdown(f"**{s.thesis}**")
                st.caption(posture)

            if company is not None:
                peer_tsr_1yr = _peer_median(peers, "total_return_1yr")
                peer_tsr_3yr = _peer_median(peers, "total_return_3yr")
                rel_1yr = (company.total_return_1yr - peer_tsr_1yr
                           if company.total_return_1yr is not None and peer_tsr_1yr is not None else None)
                rel_3yr = (company.total_return_3yr - peer_tsr_3yr
                           if company.total_return_3yr is not None and peer_tsr_3yr is not None else None)
                metric_cols = st.columns(5)
                metric_cols[0].metric("EBITDA margin", _pct(company.ebitda_margin))
                metric_cols[1].metric("Net cash / mkt cap", _pct(company.net_cash_ratio))
                metric_cols[2].metric("ROIC - cost of capital", _pct(company.roic_spread))
                metric_cols[3].metric("TSR 1yr (abs / vs. peers)", f"{_pct(company.total_return_1yr)}", _pct(rel_1yr))
                metric_cols[4].metric("TSR 3yr (abs / vs. peers)", f"{_pct(company.total_return_3yr)}", _pct(rel_3yr))
                st.caption(f"Peer group: {', '.join(p.ticker for p in peers) or 'none identified'}")

            gp = s.growth_potential
            if gp is not None:
                st.markdown("**Room for growth (computed directly from financials)**")

                def _money(v):
                    return f"${v:,.0f}" if v is not None else "n/a"

                growth_cols = st.columns(4)
                growth_cols[0].metric("Implied EBITDA uplift", _money(gp.implied_ebitda_uplift))
                growth_cols[1].metric("Implied EV upside", _pct(gp.implied_ev_upside_pct))
                growth_cols[2].metric("Implied economic-profit uplift", _money(gp.implied_economic_profit_uplift))
                growth_cols[3].metric("Balance-sheet capacity", _money(gp.balance_sheet_capacity))
                st.caption(gp.rationale)

            if show_overview_and_management:
                if s.business_overview:
                    st.markdown("**Company overview**")
                    st.write(s.business_overview)
                if s.management_assessment:
                    st.markdown("**Management assessment**")
                    st.write(s.management_assessment)

            st.markdown("**Factor scores**")
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
            if notes.get("catalyst_evidence"):
                st.markdown(f"- *Catalyst evidence:* {notes['catalyst_evidence']}")
            if notes.get("red_flags"):
                st.markdown(f"- *Red flags:* {', '.join(notes['red_flags'])}")

            if s.citations:
                st.markdown("**Sources**")
                for c in s.citations:
                    st.markdown(f"- [{c.get('title', c.get('url'))}]({c.get('url')})")

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
