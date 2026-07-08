"""
Playbook / full-report generator for shortlisted activist candidates.

Given an ActivistScore (from scorer.py) plus the underlying financials, this
calls Claude to write a consultant-style playbook: a company snapshot with
logo and 1yr/3yr TSR vs. peers, a business/management assessment aimed at
surfacing operational inefficiencies, the quantified thesis, the specific fix
an activist should push for, a staged engagement plan sized to the score
band, key risks, and a monitoring plan -- following the patterns in the
Activist Investment Candidate Selection Methodology (sections 3 and 4).

Cited sources are appended verbatim from the scoring stage's web research
(see scorer.research_and_score_qualitative) rather than left to the model to
generate, so every link in the "Sources" section is real.

This is deliberately only run for the handful of tickers a human has chosen
to investigate further (see run_screen.py) -- it's slower and more expensive
than the scoring pass, so it isn't run across the full input universe.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from scorer import ActivistScore, CompanyFinancials, ANTHROPIC_MODEL, _retry

PLAYBOOK_SYSTEM_PROMPT = """You are an activist-investment strategist writing an internal playbook for a \
portfolio manager, in the style of the firm's own methodology (derived from 19 historic activist \
campaigns: Canadian Pacific, Norfolk Southern, GE, Agrium, EMC/VMware, Arconic, Forest Labs, Hess, \
DuPont, P&G, Yahoo, Family Dollar, Baxter/Baxalta, eBay/PayPal, Nestle/L'Oreal, AOL, Apple, Microsoft, \
among others).

Core pattern to apply: the strongest campaigns pair (a) a specific, quantifiable, peer-verifiable \
performance gap with (b) a credible, already-executable fix -- a named operator with a proven track \
record, or an asset that's objectively separable. Campaigns fail or stall when either half is missing, \
or when the board can credibly rebut the fix (e.g. an integration-synergy defense).

Engagement pattern to apply: private engagement first, then a public letter with the quantified thesis, \
then formal director nominations, and only then -- if necessary -- a full proxy vote. Nearly every clean \
win was resolved by negotiated settlement before a full vote; going to a vote is where campaigns get \
expensive and outcomes get uncertain. Do not recommend maximalist, all-or-nothing terms -- partial wins \
(some board seats, voluntary adoption of part of the thesis) are a good outcome, not a failure. Size the \
position to the score band, not to conviction alone. If the fix depends on a named operator, flag key-man \
risk explicitly.

You are given a `business_overview` and `management_assessment` already drafted during scoring, plus a \
`citations` list of real sources found via web search. Use the overview/assessment as ground truth for \
the corresponding sections -- polish the prose but do not contradict them or invent new facts beyond what \
they say. Do not fabricate URLs or sources of your own; the code appends the real citation list separately.

Write in clear, direct, analyst-grade prose. Be specific to this company using the data provided -- do \
not write generic boilerplate that could apply to any company. If the data provided doesn't support a \
strong claim (e.g. no named operator was identified), say so plainly rather than inventing one.

Structure the report in this exact markdown outline (omit the '## Sources' heading -- that is appended \
by the caller, not by you):

# <Company Name> (<TICKER>) -- Activist Playbook

## Company Snapshot
(a short table of the key figures given to you -- sector, revenue, market cap, EBITDA margin, net cash
ratio, ROIC-vs-cost-of-capital spread, and 1yr/3yr TSR both in absolute terms and relative to the peer
median)

## Quantified Growth Potential
(the given growth-potential table is computed directly from financials, not estimated by you -- present
it as-is, then add 2-3 sentences interpreting what it means in plain terms: how much of the score-band
upside is margin/ROIC-driven vs. balance-sheet-driven, and how that compares to the campaigns in the
methodology's case library)

## Score Summary
(table of the five factors, total, and band/posture)

## Company Overview
(what the company does, how its business model makes money, and its position vs. peers/market --
building on the given business_overview, written so the specific inefficiency the thesis targets is
legible)

## Management Assessment
(building on the given management_assessment -- CEO/executive tenure and track record, recent
leadership/board changes, insider activity, and capital allocation discipline; be specific about where
management is or isn't the constraint)

## The Thesis
(the quantified performance gap, which peer(s) it's benchmarked against, and why the comparison holds up)

## The Fix
(the specific, executable ask -- named operator/precedent or separable asset if one exists; what the
activist should demand)

## Catalyst & Timing
(why now -- governance vulnerability, catalyst evidence, or lack thereof)

## Position Sizing & Engagement Sequence
(toehold vs. meaningful stake per the score band; the staged escalation plan: private letter -> public
letter -> director nominations -> proxy vote only if necessary; realistic settlement point)

## Key Risks & Red Flags
(structural red flags, key-man risk if applicable, and the strongest rebuttal management could credibly make)

## Monitoring Plan
(the specific, falsifiable metrics to track post-investment against the thesis)
"""


def _score_table_markdown(score: ActivistScore) -> str:
    band_label, posture = score.band
    rows = [
        ("Factor 1 -- Performance Gap", score.factor1.score, score.factor1.max_score, score.factor1.rationale),
        ("Factor 2 -- Credible Fix", score.factor2.score, score.factor2.max_score, score.factor2.rationale),
        ("Factor 3 -- Balance Sheet Slack", score.factor3.score, score.factor3.max_score, score.factor3.rationale),
        ("Factor 4 -- Catalyst/Governance", score.factor4.score, score.factor4.max_score, score.factor4.rationale),
        ("Factor 5 -- Structural Feasibility", score.factor5.score, score.factor5.max_score, score.factor5.rationale),
    ]
    lines = ["| Factor | Score | Rationale |", "|---|---|---|"]
    for label, s, m, rationale in rows:
        lines.append(f"| {label} | {s:.1f}/{m:.0f} | {rationale} |")
    lines.append(f"| **Total** | **{score.total:.1f}/100** | Band: **{band_label}** -- {posture} |")
    return "\n".join(lines)


def _peer_median(peers: list, attr: str):
    vals = [getattr(p, attr) for p in peers if getattr(p, attr) is not None]
    return statistics.median(vals) if vals else None


def _pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def _snapshot_table_markdown(financials: CompanyFinancials, peers: list) -> str:
    peer_tsr_1yr = _peer_median(peers, "total_return_1yr")
    peer_tsr_3yr = _peer_median(peers, "total_return_3yr")
    rel_1yr = (financials.total_return_1yr - peer_tsr_1yr
               if financials.total_return_1yr is not None and peer_tsr_1yr is not None else None)
    rel_3yr = (financials.total_return_3yr - peer_tsr_3yr
               if financials.total_return_3yr is not None and peer_tsr_3yr is not None else None)

    rows = [
        ("Sub-industry", financials.gics_sub_industry_name or "n/a"),
        ("Revenue", f"{financials.revenue:,.0f}" if financials.revenue else "n/a"),
        ("Market cap", f"{financials.market_cap:,.0f}" if financials.market_cap else "n/a"),
        ("EBITDA margin", _pct(financials.ebitda_margin)),
        ("Net cash / market cap", _pct(financials.net_cash_ratio)),
        ("ROIC - cost of capital", _pct(financials.roic_spread)),
        ("TSR 1yr (abs / vs. peer median)", f"{_pct(financials.total_return_1yr)} / {_pct(rel_1yr)}"),
        ("TSR 3yr (abs / vs. peer median)", f"{_pct(financials.total_return_3yr)} / {_pct(rel_3yr)}"),
        ("Peer group", ", ".join(p.ticker for p in peers) or "none identified"),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _growth_potential_table_markdown(score: ActivistScore) -> str:
    gp = score.growth_potential
    if gp is None:
        return "_Growth potential not computed for this candidate._"

    def money(v):
        return f"{v:,.0f}" if v is not None else "n/a"

    def bps(v):
        return f"{v:.0f}bps" if v is not None else "n/a"

    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "n/a"

    rows = [
        ("EBITDA margin gap vs. peer median", bps(gp.ebitda_margin_gap_bps)),
        ("Implied EBITDA uplift if gap closed", money(gp.implied_ebitda_uplift)),
        ("Implied EV upside at peer EV/EBITDA multiple", pct(gp.implied_ev_upside_pct)),
        ("ROIC-vs-cost-of-capital spread gap vs. peer median", bps(gp.roic_spread_gap_bps)),
        ("Implied annual economic-profit uplift", money(gp.implied_economic_profit_uplift)),
        ("Balance-sheet capacity vs. peer median leverage", money(gp.balance_sheet_capacity)),
    ]
    lines = ["| Growth-potential metric | Value |", "|---|---|"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    lines.append("")
    lines.append(f"_{gp.rationale}_")
    return "\n".join(lines)


def _sources_markdown(score: ActivistScore) -> str:
    if not score.citations:
        return "## Sources\n\n_No web sources were captured during research for this candidate._"
    lines = ["## Sources", ""]
    for c in score.citations:
        lines.append(f"- [{c.get('title', c.get('url'))}]({c.get('url')})")
    return "\n".join(lines)


def build_playbook_prompt(score: ActivistScore, financials: CompanyFinancials,
                           peers: list, research_notes: str = "") -> str:
    peer_summary = [
        {
            "ticker": p.ticker,
            "name": p.name,
            "ebitda_margin": p.ebitda_margin,
            "operating_ratio": p.operating_ratio,
            "roic_spread": p.roic_spread,
            "total_return_1yr": p.total_return_1yr,
            "total_return_3yr": p.total_return_3yr,
        }
        for p in peers
    ]

    gp = score.growth_potential
    payload = {
        "company": {
            "ticker": financials.ticker,
            "name": financials.name,
            "gics_sub_industry": financials.gics_sub_industry_name,
            "revenue": financials.revenue,
            "ebitda_margin": financials.ebitda_margin,
            "operating_ratio": financials.operating_ratio,
            "roic_spread": financials.roic_spread,
            "net_cash_ratio": financials.net_cash_ratio,
            "net_debt_to_ebitda": financials.net_debt_to_ebitda,
            "market_cap": financials.market_cap,
            "total_return_1yr": financials.total_return_1yr,
            "total_return_3yr": financials.total_return_3yr,
        },
        "peers": peer_summary,
        "score": score.as_row(),
        "factor_rationale": {
            "factor1": score.factor1.rationale,
            "factor2": score.factor2.rationale,
            "factor3": score.factor3.rationale,
            "factor4": score.factor4.rationale,
            "factor5": score.factor5.rationale,
        },
        "growth_potential": {
            "ebitda_margin_gap_bps": gp.ebitda_margin_gap_bps if gp else None,
            "implied_ebitda_uplift": gp.implied_ebitda_uplift if gp else None,
            "implied_ev_upside_pct": gp.implied_ev_upside_pct if gp else None,
            "roic_spread_gap_bps": gp.roic_spread_gap_bps if gp else None,
            "implied_economic_profit_uplift": gp.implied_economic_profit_uplift if gp else None,
            "balance_sheet_capacity": gp.balance_sheet_capacity if gp else None,
            "rationale": gp.rationale if gp else "",
        },
        "business_overview": score.business_overview,
        "management_assessment": score.management_assessment,
        "named_operator_or_asset": score.qualitative_notes.get("named_operator_or_asset", ""),
        "catalyst_evidence": score.qualitative_notes.get("catalyst_evidence", ""),
        "red_flags": score.qualitative_notes.get("red_flags", []),
    }

    return (
        f"Data for this candidate:\n{json.dumps(payload, indent=2, default=str)}\n\n"
        f"Additional research notes (may be empty):\n{research_notes or '(none provided)'}\n\n"
        "Write the playbook now, following the required outline exactly. Start the '## Company Snapshot' "
        f"section with this exact table:\n\n{_snapshot_table_markdown(financials, peers)}\n\n"
        f"Start the '## Quantified Growth Potential' section with this exact table:\n\n"
        f"{_growth_potential_table_markdown(score)}\n\n"
        f"Start the '## Score Summary' section with this exact table:\n\n{_score_table_markdown(score)}\n"
    )


def generate_playbook(score: ActivistScore, financials: CompanyFinancials,
                       peers: list, research_notes: str = "", anthropic_client=None) -> str:
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()

    prompt = build_playbook_prompt(score, financials, peers, research_notes)

    def do_call():
        return anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=PLAYBOOK_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

    response = _retry(do_call, what=f"Claude playbook generation for {financials.ticker}")
    text_blocks = [b.text for b in response.content if b.type == "text"]
    body = "\n".join(text_blocks).strip()

    header = f"![{financials.ticker} logo]({score.logo_url})\n\n" if score.logo_url else ""
    return f"{header}{body}\n\n{_sources_markdown(score)}\n"


def save_playbook(ticker: str, markdown_text: str, output_dir: str = "reports") -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ticker.upper()}_playbook.md"
    path.write_text(markdown_text)
    return path
