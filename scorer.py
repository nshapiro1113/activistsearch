"""
Scoring engine for the Activist Investment Candidate Selection Methodology.

Implements the five-factor, 100-point rubric (see
ActivistScoreMethodology_NedShapiro_7.7.26.pdf):

  Factor 1 - Quantifiable, Peer-Verifiable Performance Gap   (0-25, quantitative)
  Factor 2 - Credible, Executable Fix                        (0-25, LLM-assisted)
  Factor 3 - Balance Sheet / Capital Allocation Slack         (0-15, quantitative)
  Factor 4 - Catalyst / Governance Vulnerability              (0-20, LLM-assisted)
  Factor 5 - Structural Feasibility / Absence of Red Flags    (0-15, LLM-assisted, subtractive)

Factors 1 and 3 are computed deterministically from fundamentals pulled via
Capital IQ (or a supplied peer universe). Factors 2, 4 and 5 require judgment
that isn't in raw financial data, so they're drafted by an LLM (Claude) from a
structured summary of the company + its peers, and returned with rationale so
a human can review/override before capital is committed.

NOTE ON CAPITAL IQ MNEMONICS: the mnemonic strings in `CIQ_MNEMONICS` below
follow the standard S&P Capital IQ GDS ("clientservice") REST API shape, but
mnemonic availability/naming depends on your specific CIQ entitlement. Verify
each one against your account's mnemonic reference (or the Excel plug-in)
before running this at scale -- a wrong mnemonic fails loudly (empty/error
response) rather than silently, but it's still worth a spot-check on a couple
of known tickers first.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger("activistsearch.scorer")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CIQ_BASE_URL = os.environ.get(
    "CIQ_API_BASE_URL", "https://api.capitaliq.com/gdsapi/rest/v3/clientservice.json"
)
CIQ_USERNAME = os.environ.get("CIQ_USERNAME")
CIQ_PASSWORD = os.environ.get("CIQ_PASSWORD")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")

# Screening threshold from methodology section 6 ("300+ bps" gap vs peer/sector).
PERFORMANCE_GAP_THRESHOLD_BPS = 300
# Net-cash-as-%-of-market-cap that earns full Factor 3 marks (Apple-style pile).
FULL_NET_CASH_RATIO = 0.15

CIQ_MNEMONICS = {
    "revenue": "IQ_TOTAL_REV",
    "ebitda": "IQ_EBITDA",
    "ebitda_margin": "IQ_EBITDA_MARGIN",
    "operating_income": "IQ_OPER_INC",
    "net_income": "IQ_NI",
    "total_debt": "IQ_TOTAL_DEBT",
    "cash_and_st_invest": "IQ_CASH_ST_INVEST",
    "market_cap": "IQ_MARKETCAP",
    "enterprise_value": "IQ_TEV",
    "beta": "IQ_BETA",
    "interest_expense": "IQ_INT_EXPENSE",
    "total_return_1yr": "IQ_TOTAL_RETURN_1YR",
    "total_return_3yr": "IQ_TOTAL_RETURN_3YR",
    "gics_sub_industry": "IQ_GICS_SUB_INDUSTRY",
    "gics_sub_industry_name": "IQ_GICS_SUB_INDUSTRY_NAME",
}

# CAPM assumptions for the WACC proxy used in the ROIC-vs-cost-of-capital metric.
# These are blunt defaults -- override via env vars if you have house numbers.
RISK_FREE_RATE = float(os.environ.get("RISK_FREE_RATE", "0.045"))
EQUITY_RISK_PREMIUM = float(os.environ.get("EQUITY_RISK_PREMIUM", "0.05"))
DEFAULT_TAX_RATE = float(os.environ.get("DEFAULT_TAX_RATE", "0.25"))


def _retry(fn, attempts=3, base_delay=1.5, what="request"):
    last_exc = None
    for i in range(attempts):
        try:
            return fn()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = base_delay * (2 ** i)
            logger.warning("%s failed (attempt %d/%d): %s -- retrying in %.1fs",
                            what, i + 1, attempts, exc, wait)
            time.sleep(wait)
    raise last_exc


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CompanyFinancials:
    ticker: str
    name: str = ""
    revenue: Optional[float] = None
    ebitda: Optional[float] = None
    ebitda_margin: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    total_debt: Optional[float] = None
    cash_and_st_invest: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    beta: Optional[float] = None
    interest_expense: Optional[float] = None
    total_return_1yr: Optional[float] = None
    total_return_3yr: Optional[float] = None
    gics_sub_industry: Optional[str] = None
    gics_sub_industry_name: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @property
    def operating_ratio(self) -> Optional[float]:
        """Operating expense / revenue -- lower is better."""
        if self.revenue and self.operating_income is not None and self.revenue != 0:
            return (self.revenue - self.operating_income) / self.revenue
        return None

    @property
    def net_debt(self) -> Optional[float]:
        if self.total_debt is None or self.cash_and_st_invest is None:
            return None
        return self.total_debt - self.cash_and_st_invest

    @property
    def net_cash_ratio(self) -> Optional[float]:
        """(cash - debt) / market cap -- positive means a net cash pile."""
        if self.net_debt is None or not self.market_cap:
            return None
        return -self.net_debt / self.market_cap

    @property
    def net_debt_to_ebitda(self) -> Optional[float]:
        if self.net_debt is None or not self.ebitda:
            return None
        return self.net_debt / self.ebitda

    @property
    def roic(self) -> Optional[float]:
        """NOPAT / invested capital, approximated as NI+after-tax interest over (debt+equity-cash)."""
        if self.net_income is None or not self.market_cap:
            return None
        invested_capital = (self.market_cap + (self.total_debt or 0.0)
                            - (self.cash_and_st_invest or 0.0))
        if invested_capital <= 0:
            return None
        after_tax_interest = (self.interest_expense or 0.0) * (1 - DEFAULT_TAX_RATE)
        nopat = self.net_income + after_tax_interest
        return nopat / invested_capital

    @property
    def wacc(self) -> Optional[float]:
        """CAPM-based WACC proxy. Rough by design -- see module docstring."""
        if self.beta is None or not self.market_cap:
            return None
        cost_of_equity = RISK_FREE_RATE + self.beta * EQUITY_RISK_PREMIUM
        debt = self.total_debt or 0.0
        equity = self.market_cap
        total_cap = debt + equity
        if total_cap <= 0:
            return cost_of_equity
        cost_of_debt = RISK_FREE_RATE + 0.02
        if self.interest_expense and debt:
            cost_of_debt = self.interest_expense / debt
        after_tax_cost_of_debt = cost_of_debt * (1 - DEFAULT_TAX_RATE)
        return (equity / total_cap) * cost_of_equity + (debt / total_cap) * after_tax_cost_of_debt

    @property
    def roic_spread(self) -> Optional[float]:
        """ROIC minus cost of capital -- the number Factor 1 cares about."""
        if self.roic is None or self.wacc is None:
            return None
        return self.roic - self.wacc


@dataclass
class FactorScore:
    score: float
    max_score: float
    rationale: str


@dataclass
class ActivistScore:
    ticker: str
    name: str
    factor1: FactorScore
    factor2: FactorScore
    factor3: FactorScore
    factor4: FactorScore
    factor5: FactorScore
    thesis: str = ""
    business_overview: str = ""
    management_assessment: str = ""
    logo_url: Optional[str] = None
    citations: list = field(default_factory=list)
    growth_potential: Optional["GrowthPotential"] = None
    qualitative_notes: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return (self.factor1.score + self.factor2.score + self.factor3.score
                 + self.factor4.score + self.factor5.score)

    @property
    def band(self) -> tuple:
        return band_for_score(self.total)

    def as_row(self) -> dict:
        band_label, posture = self.band
        gp = self.growth_potential
        return {
            "ticker": self.ticker,
            "name": self.name,
            "thesis": self.thesis,
            "factor1_gap": round(self.factor1.score, 1),
            "factor2_fix": round(self.factor2.score, 1),
            "factor3_balance_sheet": round(self.factor3.score, 1),
            "factor4_catalyst": round(self.factor4.score, 1),
            "factor5_feasibility": round(self.factor5.score, 1),
            "total": round(self.total, 1),
            "band": band_label,
            "posture": posture,
            "implied_ev_upside_pct": (round(gp.implied_ev_upside_pct * 100, 1)
                                       if gp and gp.implied_ev_upside_pct is not None else None),
            "implied_ebitda_uplift": (round(gp.implied_ebitda_uplift, 0)
                                        if gp and gp.implied_ebitda_uplift is not None else None),
        }


SCORE_BANDS = [
    (80, 100, "High-conviction, textbook candidate",
     "Pursue aggressively; size meaningfully; expect a negotiated settlement well before a costly full vote."),
    (60, 79, "Viable, settlement-oriented candidate",
     "Pursue via staged escalation; expect a partial outcome. Avoid rigid, all-or-nothing demands."),
    (40, 59, "Speculative",
     "Toehold sizing only; expect a long, possibly multi-year, possibly multi-activist campaign -- or pass."),
    (0, 39, "Pass",
     "Thesis likely to be credibly rebutted, or the business is too opaque to underwrite the downside."),
]


def band_for_score(total: float) -> tuple:
    for lo, hi, label, posture in SCORE_BANDS:
        if lo <= total <= hi:
            return label, posture
    return "Pass", SCORE_BANDS[-1][3]


# ---------------------------------------------------------------------------
# Capital IQ client
# ---------------------------------------------------------------------------

class CapitalIQClient:
    """Thin wrapper around the CIQ GDS ('clientservice') REST API.

    Verify mnemonics in CIQ_MNEMONICS against your account before trusting
    results at scale -- see module docstring.
    """

    def __init__(self, username: str = None, password: str = None, base_url: str = None):
        self.username = username or CIQ_USERNAME
        self.password = password or CIQ_PASSWORD
        self.base_url = base_url or CIQ_BASE_URL
        if not self.username or not self.password:
            raise RuntimeError(
                "Set CIQ_USERNAME and CIQ_PASSWORD (env vars or constructor args) "
                "before using CapitalIQClient. Use MockCapitalIQClient to test "
                "the pipeline without live credentials."
            )
        self._session = requests.Session()
        self._session.auth = (self.username, self.password)

    def _gdsp(self, identifier: str, mnemonics: list) -> dict:
        body = {
            "inputRequests": [
                {"function": "GDSP", "identifier": identifier, "mnemonic": m}
                for m in mnemonics
            ]
        }

        def do_call():
            resp = self._session.post(self.base_url, json=body, timeout=30)
            resp.raise_for_status()
            return resp.json()

        data = _retry(do_call, what=f"CIQ GDSP for {identifier}")
        out = {}
        rows = data.get("GDSSDKResponse", [])
        for mnemonic, row in zip(mnemonics, rows):
            values = row.get("Rows", [])
            value = None
            if values:
                cells = values[0].get("Row", [])
                if cells:
                    value = cells[0]
            out[mnemonic] = value
        return out

    def get_fundamentals(self, ticker: str) -> CompanyFinancials:
        mnemonics = list(CIQ_MNEMONICS.values())
        raw = self._gdsp(ticker, mnemonics)
        by_field = {field_name: _to_float(raw.get(mnemonic))
                    for field_name, mnemonic in CIQ_MNEMONICS.items()}
        gics_name = raw.get(CIQ_MNEMONICS["gics_sub_industry_name"])
        return CompanyFinancials(
            ticker=ticker,
            revenue=by_field["revenue"],
            ebitda=by_field["ebitda"],
            ebitda_margin=by_field["ebitda_margin"],
            operating_income=by_field["operating_income"],
            net_income=by_field["net_income"],
            total_debt=by_field["total_debt"],
            cash_and_st_invest=by_field["cash_and_st_invest"],
            market_cap=by_field["market_cap"],
            enterprise_value=by_field["enterprise_value"],
            beta=by_field["beta"],
            interest_expense=by_field["interest_expense"],
            total_return_1yr=by_field["total_return_1yr"],
            total_return_3yr=by_field["total_return_3yr"],
            gics_sub_industry=raw.get(CIQ_MNEMONICS["gics_sub_industry"]),
            gics_sub_industry_name=gics_name if isinstance(gics_name, str) else None,
            raw=raw,
        )


def _to_float(value) -> Optional[float]:
    if value in (None, "", "NA", "N/A", "NM"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class MockCapitalIQClient:
    """Deterministic fake data so the pipeline can be smoke-tested with no
    CIQ credentials. Values are seeded off the ticker string, not random, so
    repeated runs are reproducible."""

    def get_fundamentals(self, ticker: str) -> CompanyFinancials:
        seed = sum(ord(c) for c in ticker)
        revenue = 1_000_000_000 + (seed % 50) * 200_000_000
        margin = 0.10 + (seed % 20) / 100.0
        ebitda = revenue * margin
        operating_income = ebitda * 0.8
        net_income = operating_income * 0.6
        market_cap = revenue * (1.0 + (seed % 10) / 5.0)
        total_debt = revenue * (0.1 + (seed % 7) / 20.0)
        cash = revenue * (0.05 + (seed % 5) / 20.0)
        sub_industry_bucket = seed % 6
        return CompanyFinancials(
            ticker=ticker,
            name=f"{ticker} (mock)",
            revenue=revenue,
            ebitda=ebitda,
            ebitda_margin=margin,
            operating_income=operating_income,
            net_income=net_income,
            total_debt=total_debt,
            cash_and_st_invest=cash,
            market_cap=market_cap,
            enterprise_value=market_cap + total_debt - cash,
            beta=0.8 + (seed % 8) / 10.0,
            interest_expense=total_debt * 0.05,
            total_return_1yr=-0.15 + (seed % 50) / 100.0,
            total_return_3yr=-0.1 + (seed % 40) / 100.0,
            gics_sub_industry=str(sub_industry_bucket),
            gics_sub_industry_name=f"Mock Sub-Industry {sub_industry_bucket}",
        )


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def fetch_logo_url(company_name: str) -> Optional[str]:
    """Best-effort company logo via Clearbit's public, keyless autocomplete +
    logo endpoints. Returns None on any failure -- this is cosmetic, not load
    bearing, so it never raises."""
    if not company_name:
        return None
    try:
        resp = requests.get(
            "https://autocomplete.clearbit.com/v1/companies/suggest",
            params={"query": company_name},
            timeout=5,
        )
        resp.raise_for_status()
        results = resp.json()
        if results and results[0].get("domain"):
            return f"https://logo.clearbit.com/{results[0]['domain']}"
    except (requests.RequestException, ValueError, IndexError, KeyError):
        pass
    return None


# ---------------------------------------------------------------------------
# Factor 1 - Quantifiable, Peer-Verifiable Performance Gap (0-25, quantitative)
# ---------------------------------------------------------------------------

def _peer_median(peers: list, attr: str) -> Optional[float]:
    vals = [getattr(p, attr) for p in peers if getattr(p, attr) is not None]
    return statistics.median(vals) if vals else None


def _pct_gap(candidate: Optional[float], peer: Optional[float], lower_is_better: bool) -> Optional[float]:
    """Gap in the 'opportunity' direction, in fractional terms (0.03 = 300bps).
    Positive means the candidate underperforms the peer -- i.e. there's a gap
    an activist could point to."""
    if candidate is None or peer is None:
        return None
    if lower_is_better:
        return candidate - peer
    return peer - candidate


def compute_factor1_performance_gap(company: CompanyFinancials,
                                     peers: list) -> FactorScore:
    if not peers:
        return FactorScore(5.0, 25.0, "No peer group available to benchmark against -- "
                                        "cannot verify a performance gap.")

    def peer_median(attr):
        return _peer_median(peers, attr)

    metrics = {
        "operating_ratio": (company.operating_ratio, peer_median("operating_ratio"), True),
        "ebitda_margin": (company.ebitda_margin, peer_median("ebitda_margin"), False),
        "roic_spread": (company.roic_spread, peer_median("roic_spread"), False),
        "total_return_1yr": (company.total_return_1yr, peer_median("total_return_1yr"), False),
        "total_return_3yr": (company.total_return_3yr, peer_median("total_return_3yr"), False),
    }

    gaps = {}
    for name, (cand, peer, lower_is_better) in metrics.items():
        gap = _pct_gap(cand, peer, lower_is_better)
        if gap is not None:
            gaps[name] = gap

    threshold = PERFORMANCE_GAP_THRESHOLD_BPS / 10_000.0
    material_gaps = {k: v for k, v in gaps.items() if v >= threshold}

    peer_comparability_high = len(peers) >= 2

    if not gaps:
        return FactorScore(5.0, 25.0, "Insufficient data to compute any performance-gap metric.")

    gap_desc = "; ".join(f"{k}: {v * 10_000:.0f}bps" for k, v in gaps.items())

    if len(material_gaps) >= 2 and peer_comparability_high:
        avg_gap = statistics.mean(material_gaps.values())
        score = min(25.0, 19.0 + min(6.0, avg_gap * 100))
        rationale = (f"Multiple metrics show a material gap vs. {len(peers)} peer(s) "
                     f"in {company.gics_sub_industry_name or 'the same sub-industry'} ({gap_desc}).")
    elif len(material_gaps) >= 1:
        avg_gap = statistics.mean(material_gaps.values())
        score = min(18.0, 12.0 + min(6.0, avg_gap * 100))
        rationale = (f"A real gap exists on at least one metric ({gap_desc}), but peer "
                     f"comparability is limited ({len(peers)} peer(s)) or only one metric clears "
                     f"the {PERFORMANCE_GAP_THRESHOLD_BPS}bps screening threshold.")
    else:
        best_gap = max(gaps.values())
        score = max(0.0, min(10.0, best_gap * 100))
        rationale = (f"No metric clears the {PERFORMANCE_GAP_THRESHOLD_BPS}bps threshold vs. peers "
                     f"({gap_desc}) -- company is already close to peer performance.")

    return FactorScore(round(score, 1), 25.0, rationale)


# ---------------------------------------------------------------------------
# Factor 3 - Balance Sheet / Capital Allocation Slack (0-15, quantitative)
# ---------------------------------------------------------------------------

def compute_factor3_balance_sheet_slack(company: CompanyFinancials, peers: list) -> FactorScore:
    net_cash_ratio = company.net_cash_ratio
    leverage = company.net_debt_to_ebitda
    peer_leverage_median = _peer_median(peers, "net_debt_to_ebitda")

    if net_cash_ratio is None and leverage is None:
        return FactorScore(0.0, 15.0, "Insufficient balance sheet data.")

    score = 0.0
    reasons = []

    if net_cash_ratio is not None:
        cash_component = min(15.0, max(0.0, (net_cash_ratio / FULL_NET_CASH_RATIO) * 15.0))
        score = max(score, cash_component)
        reasons.append(f"net cash is {net_cash_ratio * 100:.1f}% of market cap")

    if leverage is not None and peer_leverage_median is not None:
        underlevered_by = peer_leverage_median - leverage
        if underlevered_by > 0:
            leverage_component = min(15.0, underlevered_by * 5.0)
            score = max(score, leverage_component)
            reasons.append(
                f"levered {underlevered_by:.1f}x EBITDA below the peer median "
                f"({leverage:.1f}x vs {peer_leverage_median:.1f}x)"
            )

    score = min(15.0, score)
    rationale = ("Capital allocation slack: " + "; ".join(reasons)) if reasons else \
        "No meaningful net cash pile or leverage slack vs. peers detected."
    return FactorScore(round(score, 1), 15.0, rationale)


# ---------------------------------------------------------------------------
# Quantified growth potential -- deterministic dollar/percentage upside,
# computed directly from financials (not LLM-estimated), so the "room for
# growth" numbers in a report are exactly reproducible from the input data.
# ---------------------------------------------------------------------------

@dataclass
class GrowthPotential:
    ebitda_margin_gap_bps: Optional[float] = None
    implied_ebitda_uplift: Optional[float] = None
    implied_ev_upside_pct: Optional[float] = None
    roic_spread_gap_bps: Optional[float] = None
    implied_economic_profit_uplift: Optional[float] = None
    balance_sheet_capacity: Optional[float] = None
    rationale: str = ""


def compute_growth_potential(company: CompanyFinancials, peers: list) -> GrowthPotential:
    """Translates the peer gaps already used in Factor 1/3 into concrete
    dollar (and %) upside figures:
      - implied_ebitda_uplift: $ of EBITDA if margin matched the peer median
      - implied_ev_upside_pct: % re-rating if that uplifted EBITDA were valued
        at the peer group's own median EV/EBITDA multiple
      - implied_economic_profit_uplift: $ of annual economic profit (ROIC
        dollars above cost of capital) if the ROIC-vs-WACC spread matched peers
      - balance_sheet_capacity: $ of incremental debt capacity vs. the peer
        median leverage -- capital available for buybacks/dividends/M&A
    Each figure is left None (not zero) when there isn't enough data or the
    company already leads its peers on that metric -- absence means "no
    quantifiable gap found," not "no upside.\""""
    notes = []

    peer_ebitda_margin = _peer_median(peers, "ebitda_margin")
    ebitda_margin_gap_bps = None
    implied_ebitda_uplift = None
    if company.ebitda_margin is not None and peer_ebitda_margin is not None and company.revenue:
        gap = peer_ebitda_margin - company.ebitda_margin
        if gap > 0:
            ebitda_margin_gap_bps = gap * 10_000
            implied_ebitda_uplift = company.revenue * gap
            notes.append(f"closing the {ebitda_margin_gap_bps:.0f}bps EBITDA margin gap to the peer "
                         f"median implies ~{implied_ebitda_uplift:,.0f} of incremental annual EBITDA")

    peer_ev_ebitda_vals = [p.enterprise_value / p.ebitda for p in peers
                            if p.enterprise_value and p.ebitda]
    peer_ev_ebitda_multiple = statistics.median(peer_ev_ebitda_vals) if peer_ev_ebitda_vals else None
    implied_ev_upside_pct = None
    if (implied_ebitda_uplift and peer_ev_ebitda_multiple and company.ebitda is not None
            and company.enterprise_value):
        new_ev = (company.ebitda + implied_ebitda_uplift) * peer_ev_ebitda_multiple
        implied_ev_upside_pct = (new_ev - company.enterprise_value) / company.enterprise_value
        notes.append(f"re-rated at the peer group's {peer_ev_ebitda_multiple:.1f}x EV/EBITDA multiple, "
                     f"that implies ~{implied_ev_upside_pct * 100:.0f}% enterprise-value upside")

    peer_roic_spread = _peer_median(peers, "roic_spread")
    roic_spread_gap_bps = None
    implied_economic_profit_uplift = None
    if company.roic_spread is not None and peer_roic_spread is not None and company.market_cap:
        gap = peer_roic_spread - company.roic_spread
        if gap > 0:
            roic_spread_gap_bps = gap * 10_000
            invested_capital = (company.market_cap + (company.total_debt or 0.0)
                                 - (company.cash_and_st_invest or 0.0))
            if invested_capital > 0:
                implied_economic_profit_uplift = invested_capital * gap
                notes.append(f"closing the {roic_spread_gap_bps:.0f}bps ROIC-vs-cost-of-capital spread gap "
                             f"implies ~{implied_economic_profit_uplift:,.0f} of incremental annual economic profit")

    peer_leverage_median = _peer_median(peers, "net_debt_to_ebitda")
    balance_sheet_capacity = None
    if company.net_debt_to_ebitda is not None and peer_leverage_median is not None and company.ebitda:
        underlevered_by = peer_leverage_median - company.net_debt_to_ebitda
        if underlevered_by > 0:
            balance_sheet_capacity = underlevered_by * company.ebitda
            notes.append(f"~{balance_sheet_capacity:,.0f} of incremental debt capacity vs. the peer median "
                         f"leverage, available for buybacks/dividends/M&A")

    rationale = "; ".join(notes) if notes else \
        "No quantifiable margin, ROIC, or leverage gap vs. peers -- no clear balance-sheet or " \
        "margin-driven upside identified from financials alone."

    return GrowthPotential(
        ebitda_margin_gap_bps=ebitda_margin_gap_bps,
        implied_ebitda_uplift=implied_ebitda_uplift,
        implied_ev_upside_pct=implied_ev_upside_pct,
        roic_spread_gap_bps=roic_spread_gap_bps,
        implied_economic_profit_uplift=implied_economic_profit_uplift,
        balance_sheet_capacity=balance_sheet_capacity,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Web-search-assisted research (server tool, no beta header required)
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search"}


def _extract_citations(content_blocks) -> list:
    """Pull a deduplicated {title, url} list out of a response's content --
    both from web_search_tool_result blocks (sources the model looked at)
    and from inline citations on text blocks (sources it actually used)."""
    seen = set()
    citations = []

    def add(title, url):
        if url and url not in seen:
            seen.add(url)
            citations.append({"title": title or url, "url": url})

    for block in content_blocks:
        if block.type == "web_search_tool_result":
            results = block.content if isinstance(block.content, list) else []
            for result in results:
                add(getattr(result, "title", None), getattr(result, "url", None))
        elif block.type == "text" and getattr(block, "citations", None):
            for citation in block.citations:
                add(getattr(citation, "title", None), getattr(citation, "url", None))

    return citations


def _run_research_turn(system_prompt: str, user_content: str, anthropic_client, max_tokens=2048):
    """Turn 1 of the two-turn pattern: open-ended generation with web_search
    available (tool_choice left to its 'auto' default so Claude decides
    whether/how much to search). Returns the raw response so its content can
    be replayed verbatim into a follow-up turn that forces a structured tool."""

    def do_call():
        return anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": user_content}],
        )

    return _retry(do_call, what="Claude web-search research turn")


def _run_forced_tool_turn(prior_user_content, research_response, follow_up_instruction: str,
                           tool_schema: dict, anthropic_client, max_tokens=1536):
    """Turn 2: replay turn 1's exchange and force the structured tool call."""

    def do_call():
        return anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=max_tokens,
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_schema["name"]},
            messages=[
                {"role": "user", "content": prior_user_content},
                {"role": "assistant", "content": research_response.content},
                {"role": "user", "content": follow_up_instruction},
            ],
        )

    response = _retry(do_call, what=f"Claude forced tool call ({tool_schema['name']})")
    for block in response.content:
        if block.type == "tool_use" and block.name == tool_schema["name"]:
            return block.input
    raise RuntimeError(f"Claude did not return a tool_use block for {tool_schema['name']}")


# ---------------------------------------------------------------------------
# Peer discovery (web-search-assisted)
# ---------------------------------------------------------------------------

PEER_DISCOVERY_TOOL_SCHEMA = {
    "name": "record_peer_candidates",
    "description": "Record candidate direct-competitor tickers found via research.",
    "input_schema": {
        "type": "object",
        "properties": {
            "peers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Stock ticker, e.g. 'UNP' or 'NYSE:UNP'."},
                        "name": {"type": "string"},
                        "rationale": {"type": "string", "description": "Why this is a genuinely comparable peer -- same sub-industry, similar business model/scale."},
                    },
                    "required": ["ticker", "name", "rationale"],
                },
                "description": "3-5 of the closest publicly traded direct competitors. Empty array if none can be identified with confidence.",
            },
        },
        "required": ["peers"],
    },
}

PEER_DISCOVERY_SYSTEM_PROMPT = """You are researching direct, publicly traded competitors for a company \
being evaluated as an activist-investment candidate. The methodology requires benchmarking against a \
"near-identical peer" -- a company running a genuinely comparable business (same sub-industry, similar \
scale/geography/business model), not just the same broad sector. Use web search to find current, \
real, publicly traded competitors and their stock tickers. Do not invent tickers -- if you cannot verify \
one with reasonable confidence, leave it out. Prefer 2-4 highly comparable peers over a longer list of \
loose ones."""


def discover_peer_tickers(company: CompanyFinancials, anthropic_client=None) -> list:
    """Use web search to find real, ticker-identified direct competitors for
    `company`. Returns a list of {ticker, name, rationale} dicts (possibly
    empty). Callers are expected to fetch fundamentals for any new tickers
    and fold them into the peer group used for Factor 1/3."""
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()

    user_content = (
        f"Company: {company.name or company.ticker} (ticker: {company.ticker})\n"
        f"Sub-industry: {company.gics_sub_industry_name or 'unknown'}\n"
        f"Revenue: {company.revenue}\n\n"
        "Search the web to find its 3-5 closest publicly traded direct competitors and their tickers."
    )
    research = _run_research_turn(PEER_DISCOVERY_SYSTEM_PROMPT, user_content, anthropic_client)
    result = _run_forced_tool_turn(
        user_content, research,
        "Now record the peer candidates you found using the record_peer_candidates tool.",
        PEER_DISCOVERY_TOOL_SCHEMA, anthropic_client,
    )
    return result.get("peers", [])


# ---------------------------------------------------------------------------
# Factors 2, 4, 5 - LLM-assisted qualitative scoring
# ---------------------------------------------------------------------------

QUALITATIVE_TOOL_SCHEMA = {
    "name": "record_qualitative_scores",
    "description": "Record Factor 2, 4 and 5 scores for the activist investment rubric.",
    "input_schema": {
        "type": "object",
        "properties": {
            "one_sentence_thesis": {
                "type": "string",
                "description": "The entire activist thesis in one sentence: the quantified gap plus "
                                "the fix, e.g. 'X trades at a 900bps EBITDA margin discount to Y despite "
                                "an identical route network, and a spinoff of its logistics unit -- already "
                                "separately valued by the market -- would close most of that gap.'"
            },
            "business_overview": {
                "type": "string",
                "description": "3-5 sentences: what the company actually does, how its business model "
                                "makes money, and how it's positioned vs. its peers/market -- written so "
                                "the specific inefficiency the thesis targets is legible from the description."
            },
            "management_assessment": {
                "type": "string",
                "description": "CEO/executive tenure and track record, recent leadership or board changes, "
                                "insider buying/selling, and capital allocation decisions -- specifically "
                                "calling out where management appears to be running the business inefficiently "
                                "or where a change in leadership/discipline would unlock value. Say plainly if "
                                "the current team looks well-regarded and effective instead."
            },
            "factor2_score": {"type": "number", "description": "0-25. Credible, executable fix."},
            "factor2_rationale": {"type": "string"},
            "named_operator_or_asset": {
                "type": "string",
                "description": "Named operator with a proven precedent, or a specific separable "
                                "asset, if one exists. Empty string if none."
            },
            "factor4_score": {"type": "number", "description": "0-20. Catalyst / governance vulnerability."},
            "factor4_rationale": {"type": "string"},
            "catalyst_evidence": {"type": "string", "description": "Specific evidence cited (scandal, CEO tenure, turnover, etc.), or empty string."},
            "factor5_score": {"type": "number", "description": "0-15. Structural feasibility (start at 15, subtract for red flags)."},
            "factor5_rationale": {"type": "string"},
            "red_flags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific red flags identified, matching the methodology's red-flag categories."
            },
        },
        "required": ["one_sentence_thesis", "business_overview", "management_assessment",
                     "factor2_score", "factor2_rationale", "factor4_score",
                     "factor4_rationale", "factor5_score", "factor5_rationale", "red_flags"],
    },
}

QUALITATIVE_SYSTEM_PROMPT = """You are scoring a company as a potential activist-investment candidate \
using a fixed five-factor rubric derived from 19 historic activist campaigns. You have a web_search tool \
-- use it to ground your scoring in current, real information: recent news, management changes, insider \
transactions, governance events, and how the stock has traded relative to its peers. You will score only \
Factors 2, 4 and 5 (Factors 1 and 3 were already computed from financial data and are given to you as \
context). Be skeptical and evidence-based: if you don't have enough information to justify a high score, \
score low and say so in the rationale. Do not invent named executives or events -- if you are not confident \
one exists, leave the relevant field empty and score conservatively.

Management is the most important qualitative signal in activist investing. Specifically research and assess:
who runs the company and for how long, whether there have been recent executive or board departures/additions,
insider buying or selling, and whether recent capital allocation decisions (M&A, buybacks, capex, spinoffs)
look disciplined or wasteful. Your business_overview and management_assessment should together make clear
*where the operational or governance inefficiency is* -- that's what a real activist thesis is built on.

Factor 2 - Credible, Executable Fix (0-25):
  25 pts: a named operator with a proven track record of the exact fix elsewhere, OR an already-separable,
          separately-valued asset.
  12-18 pts: a plausible fix with no named executor, depending on incumbent management to execute it well.
  0-10 pts: the fix requires the activist to run the business itself, or has no real precedent in the sector.

Factor 4 - Catalyst / Governance Vulnerability (0-20):
  High marks: a recent scandal/forced-resignation-worthy event; a long-tenured aging CEO with no
              succession plan; unusually high executive/board turnover.
  Low marks: a well-regarded, recently-vindicated management team with strong board unity and no
             recent missteps.

Factor 5 - Structural Feasibility / Absence of Red Flags (0-15, start at 15 and subtract):
  - Genuine vertical/operational integration between segments an activist would want to separate: -5 to -10
  - Opaque legacy liabilities (financial/insurance/reserve exposure hard to diligence externally): -8 to -15
  - Dominant family/founder/government control block, or a hostile-tactics-averse shareholder base: -5 to -8
  - Extreme size where full governance capture is structurally unlikely: -3 to -5

You will also write `one_sentence_thesis`: the entire activist case in one sentence, naming the
specific quantified gap and the specific fix. If the data doesn't support a strong thesis, say so
plainly in that sentence rather than overstating it (e.g. "No clean peer gap or executable fix
identified; not currently an activist candidate.")."""


def research_and_score_qualitative(company: CompanyFinancials,
                                     factor1: FactorScore,
                                     factor3: FactorScore,
                                     research_notes: str = "",
                                     anthropic_client=None) -> tuple:
    """Two-turn web-search-assisted scoring of Factors 2, 4, 5: an open-ended
    research turn with web_search enabled, followed by a forced tool call
    that extracts the structured score. Returns (qual_dict, citations) where
    citations is a deduplicated [{title, url}, ...] list gathered from the
    research turn."""
    if anthropic_client is None:
        import anthropic
        anthropic_client = anthropic.Anthropic()

    company_summary = {
        "ticker": company.ticker,
        "name": company.name,
        "gics_sub_industry": company.gics_sub_industry_name,
        "revenue": company.revenue,
        "ebitda_margin": company.ebitda_margin,
        "market_cap": company.market_cap,
        "net_cash_ratio": company.net_cash_ratio,
        "roic_spread": company.roic_spread,
        "total_return_1yr": company.total_return_1yr,
        "total_return_3yr": company.total_return_3yr,
        "factor1_quant_score": factor1.score,
        "factor1_rationale": factor1.rationale,
        "factor3_quant_score": factor3.score,
        "factor3_rationale": factor3.rationale,
    }

    user_content = (
        f"Company financial summary:\n{json.dumps(company_summary, indent=2, default=str)}\n\n"
        f"Research notes you've been given (may be empty):\n{research_notes or '(none provided)'}\n\n"
        "Research this company: what its business does, recent news, management tenure and changes, "
        "insider activity, capital allocation track record, and how its stock has traded relative to peers "
        "over the last year. Then assess it against Factors 2, 4 and 5 of the rubric."
    )

    research = _run_research_turn(QUALITATIVE_SYSTEM_PROMPT, user_content, anthropic_client, max_tokens=3072)
    citations = _extract_citations(research.content)

    qual = _run_forced_tool_turn(
        user_content, research,
        "Now record your scores, thesis, business overview, and management assessment using the "
        "record_qualitative_scores tool.",
        QUALITATIVE_TOOL_SCHEMA, anthropic_client, max_tokens=1536,
    )
    return qual, citations


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def score_company(company: CompanyFinancials,
                    peers: list,
                    research_notes: str = "",
                    anthropic_client=None) -> ActivistScore:
    factor1 = compute_factor1_performance_gap(company, peers)
    factor3 = compute_factor3_balance_sheet_slack(company, peers)
    growth_potential = compute_growth_potential(company, peers)

    qual, citations = research_and_score_qualitative(company, factor1, factor3, research_notes, anthropic_client)
    logo_url = fetch_logo_url(company.name or company.ticker)

    factor2 = FactorScore(
        score=max(0.0, min(25.0, float(qual["factor2_score"]))),
        max_score=25.0,
        rationale=qual["factor2_rationale"],
    )
    factor4 = FactorScore(
        score=max(0.0, min(20.0, float(qual["factor4_score"]))),
        max_score=20.0,
        rationale=qual["factor4_rationale"],
    )
    factor5 = FactorScore(
        score=max(0.0, min(15.0, float(qual["factor5_score"]))),
        max_score=15.0,
        rationale=qual["factor5_rationale"],
    )

    return ActivistScore(
        ticker=company.ticker,
        name=company.name or company.ticker,
        factor1=factor1,
        factor2=factor2,
        factor3=factor3,
        factor4=factor4,
        factor5=factor5,
        thesis=qual.get("one_sentence_thesis", ""),
        business_overview=qual.get("business_overview", ""),
        management_assessment=qual.get("management_assessment", ""),
        logo_url=logo_url,
        citations=citations,
        growth_potential=growth_potential,
        qualitative_notes={
            "named_operator_or_asset": qual.get("named_operator_or_asset", ""),
            "catalyst_evidence": qual.get("catalyst_evidence", ""),
            "red_flags": qual.get("red_flags", []),
        },
    )
