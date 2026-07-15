"""About page: goal statement and the 5-factor screening methodology."""

import streamlit as st

ABOUT_GOAL = (
    "Leveraging automated research technology in the hedge fund space, activistsearch is being "
    "developed to use as a mass screener for activist opportunities: assessing companies at large "
    "scale based on activist potential."
)

CRITERIA_MARKDOWN = """
**5 factors, 2 quantitative + 3 qualitative (LLM-assisted), 100 points total.**

**Factor 1 — Quantifiable, Peer-Verifiable Performance Gap** (0–25, quantitative)
- Metrics: operating ratio (lower is better), EBITDA margin (higher is better), ROIC minus cost of
  capital i.e. "ROIC spread" (higher is better), 1yr total shareholder return vs. peers, 3yr total
  shareholder return vs. peers.
- For each metric, computes a gap in the "opportunity direction" (positive = candidate underperforms
  peers). A gap counts as *material* if it clears 300bps.
    - ≥2 material gaps + ≥2 peers: 19–25 pts (base 19, +up to 6 scaled by the average gap size)
    - ≥1 material gap (thin peer group, or only one metric clears the bar): 12–18 pts
    - No metric clears 300bps: 0–10 pts, scaled by the single best gap found
    - No peer group at all: flat 5 pts (can't verify a gap without peers)

**Factor 2 — Credible, Executable Fix** (0–25, LLM-assisted)
- 25 pts: a named operator with a proven track record of this exact fix elsewhere, OR an
  already-separable, separately-valued asset.
- 12–18 pts: a plausible fix exists but no named executor — depends on incumbent management to run
  it well.
- 0–10 pts: the activist would have to run the business itself, or there's no real precedent for this
  kind of fix in the sector.

**Factor 3 — Balance Sheet / Capital Allocation Slack** (0–15, quantitative)
- Net cash pile: `net_cash_ratio / 15% × 15` — net cash at 15% of market cap or more scores the full
  15, scaled linearly below that (calibrated to Apple's classic activist-bait cash pile).
- Under-leverage vs. peers: if net-debt/EBITDA is below the peer median, `underlevered_by × 5`,
  capped at 15 — every full turn of unused leverage capacity vs. peers is worth 5 points.
- The higher of the two components wins (not additive).

**Factor 4 — Catalyst / Governance Vulnerability** (0–20, LLM-assisted)
- High marks: a recent scandal/forced-resignation-worthy event, a long-tenured aging CEO with no
  succession plan, or unusually high executive/board turnover.
- Low marks: a well-regarded, recently-vindicated management team with strong board unity and no
  recent missteps.

**Factor 5 — Structural Feasibility / Absence of Red Flags** (0–15, LLM-assisted, subtractive)
- Starts at 15 and deducts for red flags:
    - Genuine vertical/operational integration an activist would want to separate: −5 to −10
    - Opaque legacy liabilities (insurance/reserve exposure hard to diligence externally): −8 to −15
    - Dominant family/founder/government control block, or a hostile-tactics-averse shareholder
      base: −5 to −8
    - Extreme size making governance capture structurally unlikely: −3 to −5
"""

st.header("About")
st.write(ABOUT_GOAL)
st.divider()
st.subheader("Criteria for screening (methodology)")
st.markdown(CRITERIA_MARKDOWN)
