"""LangGraph wiring for the multi-agent text-to-SQL workflow.

    START -> load_context -> planner -> generator -> validator --pass--> finalize -> END
                                ^           ^            |
                                |           +--fix_sql---+   (compile / runtime / semantic errors)
                                +-----------replan-------+   (wrong tables or approach)

CLI:
    python -m agent.graph "Top 5 product categories by revenue in 2025"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from langgraph.graph import END, START, StateGraph

from agent.mcp_client import WarehouseTools
from agent.nodes import finalize, generator, load_context, planner, route_after_validation, validator
from agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_context", load_context)
    g.add_node("planner", planner)
    g.add_node("generator", generator)
    g.add_node("validator", validator)
    g.add_node("finalize", finalize)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "planner")
    g.add_edge("planner", "generator")
    g.add_edge("generator", "validator")
    g.add_conditional_edges(
        "validator",
        route_after_validation,
        {"finalize": "finalize", "generator": "generator", "planner": "planner"},
    )
    g.add_edge("finalize", END)
    return g.compile()


GRAPH = build_graph()


async def run_question(question: str, *, options: dict | None = None, llm=None, tools: WarehouseTools | None = None) -> dict:
    """Answer one question. Opens its own MCP session / LLM client unless given."""
    from agent.llm import LLM

    llm = llm or LLM()
    calls_before = llm.usage.calls
    start = time.perf_counter()

    async def _run(t: WarehouseTools) -> dict:
        config = {"configurable": {"tools": t, "llm": llm, "options": options or {}}, "recursion_limit": 60}
        return await GRAPH.ainvoke({"question": question, "history": [], "trace": []}, config=config)

    if tools is not None:
        final = await _run(tools)
    else:
        async with WarehouseTools() as t:
            final = await _run(t)

    final["llm_calls"] = llm.usage.calls - calls_before
    final["elapsed_s"] = round(time.perf_counter() - start, 2)
    return final


def ask(question: str, **kwargs) -> dict:
    """Synchronous convenience wrapper (used by the Streamlit app)."""
    return asyncio.run(run_question(question, **kwargs))


def mermaid() -> str:
    return GRAPH.get_graph().draw_mermaid()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--no-semantic", action="store_true", help="disable the LLM semantic validator")
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()
    out = ask(args.question, options={"semantic_check": not args.no_semantic, "summarize": True})
    if args.trace:
        for ev in out.get("trace", []):
            print(json.dumps(ev, default=str))
        print()
    print(f"Status: {out.get('status')}  attempts={out.get('attempts')}  llm_calls={out['llm_calls']}  {out['elapsed_s']}s")
    print(f"\nSQL:\n{out.get('sql')}\n")
    res = out.get("result") or {}
    if res.get("columns"):
        print(" | ".join(res["columns"]))
        for row in res["rows"][:20]:
            print(" | ".join(str(v) for v in row))
    if out.get("answer"):
        print(f"\nAnswer: {out['answer']}")


if __name__ == "__main__":
    main()
