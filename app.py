"""
Streamlit web UI for the activist-candidate screener.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

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

page = st.navigation([
    st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/query_stats:"),
    st.Page("app_pages/about.py", title="About", icon=":material/info:"),
])
page.run()
