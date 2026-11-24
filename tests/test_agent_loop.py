"""Exercise the agents' self-correction loop against the real MCP server with a scripted LLM.

The `drive` helper mirrors the LangGraph wiring in agent/graph.py, so the node
logic and routing are tested even where LangGraph isn't installed. If LangGraph
is installed, the real compiled graph is tested too.
"""

import asyncio
import json

import pytest

from agent.llm import Usage, parse_json
from agent.mcp_client import WarehouseTools
from agent.nodes import finalize, generator, load_context, planner, route_after_validation, validator

PLAN = {"tables": ["orders", "order_items", "categories", "products", "made_up_table"],
        "joins": [], "filters": [], "metrics": [], "group_by": [], "order_and_limit": "",
        "value_lookups": [{"table": "orders", "column": "status"}], "assumptions": []}

BAD_COLUMN = "SELECT SUM(revenue) FROM orders"
NO_CANCEL_FILTER = ("SELECT SUM(oi.quantity*oi.unit_price*(1-oi.discount)) AS revenue FROM order_items oi "
                    "JOIN orders o ON o.order_id = oi.order_id WHERE o.order_date >= '2024-01-01' AND o.order_date < '2025-01-01'")
CORRECT = NO_CANCEL_FILTER + " AND o.status <> 'Cancelled'"


class FakeLLM:
    def __init__(self, script: dict[str, list[str]]):
        self.script = {k: list(v) for k, v in script.items()}
        self.usage = Usage()
        self.prompts: dict[str, list[str]] = {}

    async def complete(self, system, user, *, agent="llm"):
        self.usage.calls += 1
        self.prompts.setdefault(agent, []).append(user)
        return self.script[agent].pop(0)

    async def complete_json(self, system, user, *, agent="llm"):
        return parse_json(await self.complete(system, user, agent=agent))


def merge(state, update):
    for k, v in update.items():
        state[k] = state.get(k, []) + v if k in ("history", "trace") else v


async def drive(question, llm, options=None):
    async with WarehouseTools() as tools:
        config = {"configurable": {"tools": tools, "llm": llm, "options": options or {}}}
        state = {"question": question, "history": [], "trace": []}
        merge(state, await load_context(state, config))
        nxt = "planner"
        while nxt != "finalize":
            if nxt == "planner":
                merge(state, await planner(state, config))
            merge(state, await generator(state, config))
            merge(state, await validator(state, config))
            nxt = route_after_validation(state, config)
        merge(state, await finalize(state, config))
        return state


def sql_block(sql):
    return f"```sql\n{sql}\n```"


def test_self_correction_compile_then_semantic():
    llm = FakeLLM({
        "planner": [json.dumps(PLAN)],
        "generator": [sql_block(BAD_COLUMN), sql_block(NO_CANCEL_FILTER), sql_block(CORRECT)],
        "validator": [
            '{"verdict": "fix_sql", "issues": ["Cancelled orders are included"], "suggestion": "Add status <> \'Cancelled\'"}',
            '```json\n{"verdict": "pass", "issues": [], "suggestion": ""}\n```',
        ],
    })
    s = asyncio.run(drive("What was total revenue in 2024?", llm))

    assert s["status"] == "success"
    assert s["attempts"] == 3
    assert s["sql"] == CORRECT
    assert [h["stage"] for h in s["history"]] == ["compile", "semantic"]
    assert abs(s["result"]["rows"][0][0] - 1997264.24) < 0.01
    # planner output was grounded: unknown table dropped, real status values looked up
    assert "made_up_table" not in s["plan"]["tables"]
    assert "Cancelled" in s["schema_context"]
    # generator saw the earlier failures on retries
    assert "no such column" in llm.prompts["generator"][1]
    assert "Cancelled orders are included" in llm.prompts["generator"][2]
    assert s["data_end"] == "2025-12-31"


def test_replan_route():
    llm = FakeLLM({
        "planner": [json.dumps({**PLAN, "tables": ["payments"]}), json.dumps(PLAN)],
        "generator": [sql_block("SELECT SUM(amount) FROM payments"), sql_block(CORRECT)],
        "validator": ['{"verdict": "replan", "issues": ["Revenue must come from order_items"], "suggestion": "Use order_items"}',
                      '{"verdict": "pass", "issues": [], "suggestion": ""}'],
    })
    s = asyncio.run(drive("What was total revenue in 2024?", llm))
    assert s["status"] == "success" and s["replans"] == 1
    assert [e["title"] for e in s["trace"] if e["agent"] == "planner"] == ["Planned", "Re-planned"]


def test_gives_up_after_max_attempts_but_returns_last_runnable_query():
    llm = FakeLLM({
        "planner": [json.dumps(PLAN)],
        "generator": [sql_block(NO_CANCEL_FILTER), sql_block(BAD_COLUMN)],
        "validator": ['{"verdict": "fix_sql", "issues": ["x"], "suggestion": "x"}'],
    })
    s = asyncio.run(drive("What was total revenue in 2024?", llm, {"max_attempts": 2}))
    assert s["status"] == "failed"
    assert s["attempts"] == 2
    assert s["sql"] == NO_CANCEL_FILTER  # best effort: last query that executed
    assert s["result"]["row_count"] == 1


def test_blocks_destructive_sql_from_llm():
    llm = FakeLLM({
        "planner": [json.dumps(PLAN)],
        "generator": [sql_block("DELETE FROM orders"), sql_block(CORRECT)],
        "validator": ['{"verdict": "pass", "issues": [], "suggestion": ""}'],
    })
    s = asyncio.run(drive("delete all orders", llm))
    assert "read-only" in s["history"][0]["feedback"]


def test_semantic_check_can_be_disabled():
    llm = FakeLLM({"planner": [json.dumps(PLAN)], "generator": [sql_block(NO_CANCEL_FILTER)], "validator": []})
    s = asyncio.run(drive("What was total revenue in 2024?", llm, {"semantic_check": False}))
    assert s["status"] == "success" and s["attempts"] == 1 and "validator" not in llm.prompts


def test_real_langgraph_graph():
    pytest.importorskip("langgraph")
    from agent.graph import run_question

    llm = FakeLLM({
        "planner": [json.dumps(PLAN)],
        "generator": [sql_block(BAD_COLUMN), sql_block(CORRECT)],
        "validator": ['{"verdict": "pass", "issues": [], "suggestion": ""}'],
    })
    out = asyncio.run(run_question("What was total revenue in 2024?", llm=llm))
    assert out["status"] == "success" and out["attempts"] == 2 and out["llm_calls"] == 4
