"""
Entry point: score a user-supplied list of tickers, show the ranked results,
then interactively ask which candidates to investigate further -- and only
generate full playbooks/reports for those.

Usage:
    python run_screen.py AAPL MSFT CP CNI NSC UNP
    python run_screen.py --tickers-file tickers.csv
    python run_screen.py --mock AAPL MSFT CP CNI          # no CIQ/Anthropic creds needed
    python run_screen.py --tickers-file tickers.csv --notes-file notes.json --non-interactive TICKER1,TICKER2

Peer groups are drawn first from *within* the tickers you supply (grouped by
GICS sub-industry); for any candidate with fewer than 2 in-list peers, Claude
is asked to find real, ticker-identified direct competitors via web search,
and their fundamentals are pulled in automatically to fill out the peer group
used for Factor 1/3.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import anthropic

from scorer import (
    CapitalIQClient,
    FreeDataClient,
    MockCapitalIQClient,
    discover_peer_tickers,
    score_company,
)
from playbook import generate_playbook, save_playbook

MIN_PEERS_BEFORE_DISCOVERY = 2


def _normalize_ticker(raw: str) -> str:
    """Strip an 'EXCHANGE:TICKER' prefix if present, e.g. 'NYSE:UNP' -> 'UNP'."""
    return raw.split(":")[-1].strip().upper()


def augment_peers_via_web_search(financials_by_ticker: dict, peer_groups: dict,
                                   ciq_client, anthropic_client) -> None:
    """Mutates financials_by_ticker and peer_groups in place: for any company
    with too few in-list peers, discovers real competitors via web search and
    pulls in their fundamentals."""
    for ticker, company in list(financials_by_ticker.items()):
        if len(peer_groups.get(ticker, [])) >= MIN_PEERS_BEFORE_DISCOVERY:
            continue
        print(f"  {ticker}: only {len(peer_groups.get(ticker, []))} in-list peer(s) -- "
              f"searching the web for direct competitors...")
        try:
            candidates = discover_peer_tickers(company, anthropic_client=anthropic_client)
        except Exception as exc:
            print(f"    WARNING: peer discovery failed for {ticker}: {exc}", file=sys.stderr)
            continue

        for candidate in candidates:
            # Claude's forced tool call is supposed to return {ticker, name,
            # rationale} objects per PEER_DISCOVERY_TOOL_SCHEMA, but tool
            # schemas aren't strictly guaranteed -- it can occasionally
            # return a bare ticker string instead. Handle both shapes.
            if isinstance(candidate, dict):
                raw_ticker = candidate.get("ticker", "")
                rationale = candidate.get("rationale", "")
            else:
                raw_ticker = str(candidate)
                rationale = ""
            peer_ticker = _normalize_ticker(raw_ticker)
            if not peer_ticker or peer_ticker == ticker:
                continue
            if peer_ticker not in financials_by_ticker:
                try:
                    financials_by_ticker[peer_ticker] = ciq_client.get_fundamentals(peer_ticker)
                except Exception as exc:
                    print(f"    WARNING: could not fetch discovered peer {peer_ticker}: {exc}", file=sys.stderr)
                    continue
            peer_company = financials_by_ticker[peer_ticker]
            if peer_company not in peer_groups[ticker]:
                peer_groups[ticker].append(peer_company)
                print(f"    + added {peer_ticker} ({rationale[:80]})")


def read_tickers_file(path: str) -> list:
    tickers = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            candidate = row[0].strip().upper()
            if candidate and not candidate.startswith("#") and candidate != "TICKER":
                tickers.append(candidate)
    return tickers


def load_research_notes(path: str) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def build_peer_groups(financials_by_ticker: dict) -> dict:
    """Group companies by GICS sub-industry within the supplied universe.
    Returns {ticker: [peer CompanyFinancials, ...]} (self excluded)."""
    by_sub_industry = defaultdict(list)
    for company in financials_by_ticker.values():
        key = company.gics_sub_industry or "UNKNOWN"
        by_sub_industry[key].append(company)

    peer_groups = {}
    for ticker, company in financials_by_ticker.items():
        key = company.gics_sub_industry or "UNKNOWN"
        peer_groups[ticker] = [c for c in by_sub_industry[key] if c.ticker != ticker]
    return peer_groups


def print_results_table(rows: list) -> None:
    headers = ["ticker", "total", "band", "factor1_gap", "factor2_fix",
               "factor3_balance_sheet", "factor4_catalyst", "factor5_feasibility"]
    widths = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))
    for r in rows:
        print("  ".join(str(r[h]).ljust(widths[h]) for h in headers))

    print("\nThesis, one sentence per opportunity:")
    for r in rows:
        print(f"  [{r['total']:.1f}] {r['ticker']}: {r['thesis']}")


def write_results_csv(rows: list, path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def prompt_for_selection(tickers: list) -> list:
    print(
        "\nWhich tickers would you like full playbooks/reports for? "
        "Enter comma-separated tickers, 'all', or 'none':"
    )
    raw = input("> ").strip()
    if not raw or raw.lower() == "none":
        return []
    if raw.lower() == "all":
        return tickers
    chosen = [t.strip().upper() for t in raw.split(",") if t.strip()]
    valid = [t for t in chosen if t in tickers]
    invalid = [t for t in chosen if t not in tickers]
    if invalid:
        print(f"Ignoring unrecognized tickers: {', '.join(invalid)}")
    return valid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="*", help="Tickers to score, space-separated.")
    parser.add_argument("--tickers-file", help="CSV/text file with one ticker per line.")
    parser.add_argument("--notes-file", help="JSON file mapping ticker -> research notes string, "
                                              "fed to the LLM scoring and playbook steps.")
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--mock", action="store_true",
                              help="Use synthetic financial data instead of live Capital IQ "
                                   "(still calls the real Anthropic API unless ANTHROPIC_API_KEY is unset).")
    data_source.add_argument("--free-data", action="store_true",
                              help="Use free fundamentals from SEC EDGAR (XBRL filings) + Yahoo Finance "
                                   "instead of Capital IQ. No credentials needed, but only covers US SEC "
                                   "filers and uses SIC codes instead of GICS for peer grouping.")
    parser.add_argument("--output-csv", default="scores.csv", help="Where to write the ranked score table.")
    parser.add_argument("--reports-dir", default="reports", help="Directory for generated playbooks.")
    parser.add_argument("--non-interactive", metavar="TICKER1,TICKER2",
                         help="Skip the interactive prompt and generate playbooks for exactly these "
                              "tickers (comma-separated). Pass an empty string to skip playbooks entirely.")
    args = parser.parse_args()

    tickers = list(args.tickers)
    if args.tickers_file:
        tickers.extend(read_tickers_file(args.tickers_file))
    tickers = list(dict.fromkeys(t.upper() for t in tickers))  # de-dupe, preserve order

    if not tickers:
        parser.error("No tickers supplied. Pass them as arguments or via --tickers-file.")

    research_notes = load_research_notes(args.notes_file)

    if args.mock:
        ciq_client = MockCapitalIQClient()
    elif args.free_data:
        ciq_client = FreeDataClient()
    else:
        ciq_client = CapitalIQClient()
    anthropic_client = anthropic.Anthropic()

    print(f"Fetching fundamentals for {len(tickers)} ticker(s)...")
    financials_by_ticker = {}
    for ticker in tickers:
        try:
            financials_by_ticker[ticker] = ciq_client.get_fundamentals(ticker)
        except Exception as exc:
            print(f"  WARNING: failed to fetch {ticker}, skipping: {exc}", file=sys.stderr)

    if not financials_by_ticker:
        print("No fundamentals could be fetched for any ticker. Aborting.", file=sys.stderr)
        sys.exit(1)

    peer_groups = build_peer_groups(financials_by_ticker)

    print("Checking peer coverage (finding direct competitors via web search where needed)...")
    augment_peers_via_web_search(financials_by_ticker, peer_groups, ciq_client, anthropic_client)

    print("Scoring candidates (Factors 1 & 3 computed, Factors 2/4/5 drafted by Claude)...")
    scores = {}
    for ticker in tickers:
        if ticker not in financials_by_ticker:
            continue
        company = financials_by_ticker[ticker]
        try:
            scores[ticker] = score_company(
                company,
                peer_groups[ticker],
                research_notes=research_notes.get(ticker, ""),
                anthropic_client=anthropic_client,
            )
        except Exception as exc:
            print(f"  WARNING: failed to score {ticker}, skipping: {exc}", file=sys.stderr)

    if not scores:
        print("No candidates could be scored. Aborting.", file=sys.stderr)
        sys.exit(1)

    ranked = sorted(scores.values(), key=lambda s: s.total, reverse=True)
    rows = [s.as_row() for s in ranked]

    print()
    print_results_table(rows)
    write_results_csv(rows, args.output_csv)
    print(f"\nFull score table written to {args.output_csv}")

    scored_tickers = [s.ticker for s in ranked]
    if args.non_interactive is not None:
        selected = [t.strip().upper() for t in args.non_interactive.split(",") if t.strip()]
        selected = [t for t in selected if t in scored_tickers]
    else:
        selected = prompt_for_selection(scored_tickers)

    if not selected:
        print("\nNo tickers selected for a full playbook/report. Done.")
        return

    print(f"\nGenerating full playbooks/reports for: {', '.join(selected)}")
    for ticker in selected:
        score = scores[ticker]
        company = financials_by_ticker[ticker]
        peers = peer_groups[ticker]
        try:
            markdown_text = generate_playbook(
                score, company, peers, research_notes=research_notes.get(ticker, ""),
                anthropic_client=anthropic_client,
            )
            path = save_playbook(ticker, markdown_text, output_dir=args.reports_dir)
            print(f"  {ticker}: wrote {path}")
        except Exception as exc:
            print(f"  WARNING: failed to generate playbook for {ticker}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
