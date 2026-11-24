"""Streamlit UI for the agentic text-to-SQL assistant.

    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from config import settings

st.set_page_config(page_title="Text-to-SQL Analytics Assistant", page_icon="🔎", layout="wide")

EXAMPLES = [
    "What are the top 5 product categories by revenue in 2025?",
    "Show monthly revenue for 2025",
    "For each customer region, show 2024 revenue, 2025 revenue and growth %",
    "Which carrier has the fastest average delivery time?",
    "What percentage of order line items were returned in each category?",
    "What was revenue last quarter?",
]
AGENT_ICONS = {"system": "⚙️", "planner": "🧭", "generator": "✍️", "validator": "🛡️"}


# --------------------------------------------------------------------------- #
def ensure_warehouse() -> None:
    if settings.backend == "sqlite" and not Path(settings.sqlite_path).exists():
        from warehouse.seed import build_sqlite

        with st.spinner("Building local warehouse (first run only)..."):
            build_sqlite(Path(settings.sqlite_path))


def auto_chart(df: pd.DataFrame) -> None:
    """Draw a simple chart when the result shape obviously suits one."""
    if df.empty or len(df) < 2 or len(df) > 60 or df.shape[1] < 2:
        return
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    labels = [c for c in df.columns if c not in numeric]
    if not numeric or not labels:
        return
    x, ys = labels[0], numeric[:3]
    chart_df = df.set_index(x)[ys]
    looks_temporal = df[x].astype(str).str.match(r"^\d{4}(-\d{2})?(-\d{2})?$").all()
    if looks_temporal:
        st.line_chart(chart_df)
    else:
        st.bar_chart(chart_df)


def render_trace(trace: list[dict]) -> None:
    for ev in trace:
        agent = ev.get("agent", "system")
        st.markdown(f"**{AGENT_ICONS.get(agent, '•')} {agent.title()}: {ev.get('title', '')}**")
        if "plan" in ev:
            st.json(ev["plan"], expanded=False)
            if ev.get("value_lookups"):
                st.caption("Grounded values: " + " · ".join(ev["value_lookups"]))
        if "sql" in ev:
            st.code(ev["sql"], language="sql")
        if "feedback" in ev:
            st.warning(f"{ev['feedback']}  \n→ routed back to **{ev.get('route')}**")
        if "rows" in ev:
            st.caption(f"Approved, {ev['rows']} row(s)")


def render_answer(out: dict) -> None:
    if out.get("error"):
        st.error(out["error"])
        return
    status = out.get("status")
    res = out.get("result") or {}
    if status != "success":
        st.warning("The validator didn't fully approve any query within the attempt limit. "
                   "Showing the last query that ran; double-check it.")
    if out.get("answer"):
        st.markdown(out["answer"])
    if res.get("columns"):
        df = pd.DataFrame(res["rows"], columns=res["columns"])
        st.dataframe(df, hide_index=True)
        if res.get("truncated"):
            st.caption(f"Showing first {res['row_count']} rows.")
        auto_chart(df)
    elif res.get("error"):
        st.error(res["error"])
    if out.get("sql"):
        with st.expander("SQL", expanded=False):
            st.code(out["sql"], language="sql")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "✅ validated" if status == "success" else "⚠️ unverified")
    c2.metric("SQL attempts", out.get("attempts", 0))
    c3.metric("LLM calls", out.get("llm_calls", 0))
    c4.metric("Time", f"{out.get('elapsed_s', 0)}s")
    with st.expander("Agent trace (planner → generator → validator)"):
        render_trace(out.get("trace", []))


# --------------------------------------------------------------------------- #
ensure_warehouse()

with st.sidebar:
    st.header("Settings")
    st.caption(f"Warehouse: **{settings.backend}**")
    st.caption(f"LLM: **{settings.llm_provider}**")
    default_model = {"ollama": settings.ollama_model, "groq": settings.groq_model,
                     "gemini": settings.gemini_model}.get(settings.llm_provider, settings.anthropic_model)
    model = st.text_input("Model", default_model)
    max_attempts = st.slider("Max SQL attempts", 1, 6, settings.max_attempts)
    semantic = st.toggle("Semantic validator", value=True,
                         help="LLM reviews results for wrong filters, fan-out, etc. Off = only compile/runtime checks.")
    summarize = st.toggle("Natural-language answer", value=True)
    st.divider()
    st.subheader("Try an example")
    for ex in EXAMPLES:
        if st.button(ex):
            st.session_state["pending"] = ex
    st.divider()
    if st.button("Clear chat"):
        st.session_state["messages"] = []

st.title("🔎 Text-to-SQL Analytics Assistant")
st.caption("Ask about the 12-table retail warehouse in plain English. A planner, a SQL generator and a "
           "validator agent work together over MCP, retrying until the query checks out.")

if settings.llm_provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY is not set. Add it to your .env file and restart.")
    st.stop()

st.session_state.setdefault("messages", [])
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["content"])

question = st.chat_input("e.g. Which store had the highest in-store revenue in 2025?")
question = question or st.session_state.pop("pending", None)

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Planning, writing and validating SQL..."):
            try:
                from agent.graph import ask
                from agent.llm import LLM

                out = ask(question, llm=LLM(model=model),
                          options={"max_attempts": max_attempts, "semantic_check": semantic, "summarize": summarize})
                out = json.loads(json.dumps(out, default=str))  # make it safely storable
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}"}
        render_answer(out)
    st.session_state["messages"].append({"role": "assistant", "content": out})
