"""Agent nodes: context loader, planner, generator, validator, finalizer.

Each node is `async def node(state, config) -> partial_state`. Dependencies come
from config["configurable"]:
    tools   : agent.mcp_client.WarehouseTools (MCP session)
    llm     : agent.llm.LLM
    options : {"max_attempts", "max_replans", "semantic_check", "summarize", "preview_rows"}
"""

from __future__ import annotations

import json
from typing import Any

from agent import prompts as P
from agent.llm import extract_sql
from agent.state import AgentState, event
from config import settings


try:
    from langchain_core.runnables import RunnableConfig
except ImportError:  # lets the nodes be unit-tested without LangGraph installed
    RunnableConfig = dict  # type: ignore[misc,assignment]


def _opts(config: dict | None) -> dict:
    c = (config or {}).get("configurable", {})
    return {
        "max_attempts": settings.max_attempts,
        "max_replans": 1,
        "semantic_check": True,
        "summarize": False,
        "preview_rows": settings.preview_rows,
        **c.get("options", {}),
    }


def _deps(config: dict) -> tuple[Any, Any, dict]:
    c = (config or {}).get("configurable", {})
    return c["tools"], c.get("llm"), _opts(config)


def _rules(state: AgentState) -> str:
    return P.COMMON_RULES.format(data_start=state.get("data_start", "?"), data_end=state.get("data_end", "?"))


def _preview(result: dict, n: int) -> str:
    cols = result.get("columns", [])
    lines = [" | ".join(cols)] if cols else []
    for row in result.get("rows", [])[:n]:
        lines.append(" | ".join("NULL" if v is None else str(v) for v in row))
    return "\n".join(lines) or "(no rows)"


# --------------------------------------------------------------------------- #
async def load_context(state: AgentState, config: RunnableConfig) -> dict:
    tools, _llm, _ = _deps(config)
    dialect = (await tools.call("get_dialect"))["dialect"]
    tables = (await tools.call("list_tables"))["tables"]
    catalog = "\n".join(f"- {t['table']} ({t['row_count']:,} rows): {t['description']}"
                        if t.get("row_count") is not None else f"- {t['table']}: {t['description']}"
                        for t in tables)
    rng = await tools.call("run_query", sql="SELECT MIN(order_date), MAX(order_date) FROM orders", limit=1)
    start, end = (rng["rows"][0] if rng.get("rows") else ["unknown", "unknown"])
    return {
        "dialect": dialect, "catalog": catalog, "data_start": str(start), "data_end": str(end),
        "attempts": 0, "replans": 0,
        "trace": [event("system", "Loaded warehouse context", dialect=dialect, tables=len(tables),
                        data_range=f"{start} to {end}")],
    }


# --------------------------------------------------------------------------- #
async def planner(state: AgentState, config: RunnableConfig) -> dict:
    tools, llm, _ = _deps(config)
    feedback = ""
    is_replan = bool(state.get("plan"))
    if is_replan:
        v = state.get("validation", {})
        feedback = (
            "\nA previous plan led to an incorrect query. Fix the approach.\n"
            f"Previous plan: {json.dumps(state['plan'])}\n"
            f"Last SQL:\n{state.get('sql', '')}\n"
            f"Validator issues: {v.get('issues')}\nSuggestion: {v.get('suggestion', '')}"
        )
    plan = await llm.complete_json(
        P.PLANNER_SYSTEM.format(dialect=state["dialect"], rules=_rules(state)),
        P.PLANNER_USER.format(question=state["question"], catalog=state["catalog"], feedback=feedback),
        agent="planner",
    )

    # Ground the plan with real schema details via MCP
    known = {line.split(" ")[1] for line in state["catalog"].splitlines() if line.startswith("- ")}
    tables = [t for t in dict.fromkeys(str(x).lower() for x in plan.get("tables", [])) if t in known]
    dropped = [t for t in plan.get("tables", []) if str(t).lower() not in known]
    if not tables:  # planner produced nothing usable: give the generator everything
        tables = sorted(known)
    plan["tables"] = tables

    parts = []
    for t in tables:
        d = await tools.call("describe_table", table=t)
        cols = "\n".join(
            f"    {c['name']} {c['type']}  -- {c['description']}"
            + (f" (references {c['references']})" if c.get("references") else "")
            for c in d.get("columns", [])
        )
        parts.append(f"TABLE {t}: {d.get('description', '')}\n{cols}")
    lookups = []
    for lk in (plan.get("value_lookups") or [])[:4]:
        try:
            vals = await tools.call("column_values", table=lk["table"], column=lk["column"], limit=15)
            if not vals.get("error"):
                lookups.append(f"{lk['table']}.{lk['column']} values: " +
                               ", ".join(repr(r[0]) for r in vals.get("rows", [])))
        except Exception:  # noqa: BLE001 - lookups are best-effort hints
            continue
    schema = "\n\n".join(parts) + ("\n\nKnown column values:\n" + "\n".join(lookups) if lookups else "")

    return {
        "plan": plan,
        "schema_context": schema,
        "replans": state.get("replans", 0) + (1 if is_replan else 0),
        "trace": [event("planner", "Re-planned" if is_replan else "Planned", plan=plan,
                        dropped_unknown_tables=dropped, value_lookups=lookups)],
    }


# --------------------------------------------------------------------------- #
async def generator(state: AgentState, config: RunnableConfig) -> dict:
    _tools, llm, opts = _deps(config)
    history = ""
    if state.get("history"):
        recent = state["history"][-3:]
        history = "\nPrevious attempts failed. Do not repeat these mistakes:\n" + "\n".join(
            f"--- Attempt {h['attempt']} ({h['stage']}) ---\n{h['sql']}\nProblem: {h['feedback']}" for h in recent
        )
    text = await llm.complete(
        P.GENERATOR_SYSTEM.format(dialect=state["dialect"], rules=_rules(state),
                                  dialect_notes=P.DIALECT_NOTES.get(state["dialect"], "")),
        P.GENERATOR_USER.format(question=state["question"], plan=json.dumps(state.get("plan", {}), indent=2),
                                schema=state.get("schema_context", ""), history=history),
        agent="generator",
    )
    sql = extract_sql(text)
    attempt = state.get("attempts", 0) + 1
    return {"sql": sql, "attempts": attempt, "trace": [event("generator", f"Generated SQL (attempt {attempt})", sql=sql)]}


# --------------------------------------------------------------------------- #
async def validator(state: AgentState, config: RunnableConfig) -> dict:
    """Three gates: (1) safety + compile via EXPLAIN, (2) execution, (3) LLM semantic review."""
    tools, llm, opts = _deps(config)
    sql, attempt = state["sql"], state["attempts"]

    def fail(stage: str, feedback: str, route: str, issues: list[str] | None = None, **extra) -> dict:
        out = {
            "validation": {"ok": False, "stage": stage, "issues": issues or [feedback], "suggestion": feedback, "route": route},
            "history": [{"attempt": attempt, "sql": sql, "stage": stage, "feedback": feedback}],
            "trace": [event("validator", f"Rejected at {stage} check", feedback=feedback, route=route)],
        }
        out.update(extra)
        return out

    # Gate 1: read-only guard + compile
    check = await tools.call("validate_sql", sql=sql)
    if not check.get("valid"):
        return fail("compile", f"Query failed to compile: {check.get('error')}", "generator")

    # Gate 2: execute
    result = await tools.call("run_query", sql=sql, limit=settings.result_row_limit)
    if result.get("error"):
        return fail("execution", f"Query failed at runtime: {result['error']}", "generator")
    last_good = {"sql": sql, "result": result}

    # Gate 3: semantic review
    if opts["semantic_check"] and llm is not None:
        review = await llm.complete_json(
            P.VALIDATOR_SYSTEM.format(dialect=state["dialect"], rules=_rules(state)),
            P.VALIDATOR_USER.format(
                question=state["question"], plan=json.dumps(state.get("plan", {}), indent=2), sql=sql,
                row_count=result["row_count"], truncated=" (truncated)" if result.get("truncated") else "",
                columns=result["columns"], preview=_preview(result, opts["preview_rows"]),
            ),
            agent="validator",
        )
        verdict = str(review.get("verdict", "pass")).lower()
        if verdict in ("fix_sql", "replan"):
            route = "planner" if verdict == "replan" else "generator"
            issues = [str(i) for i in (review.get("issues") or [])]
            suggestion = str(review.get("suggestion") or "")
            feedback = "Issues: " + "; ".join(issues) + (f". Fix: {suggestion}" if suggestion else "")
            return fail("semantic", feedback, route, issues=issues,
                        result=result, last_good=last_good)

    return {
        "validation": {"ok": True, "stage": "passed", "issues": [], "route": "finalize"},
        "result": result,
        "last_good": last_good,
        "trace": [event("validator", "Approved", rows=result["row_count"])],
    }


def route_after_validation(state: AgentState, config: RunnableConfig = None) -> str:
    """Self-correction routing: finalize, back to the generator, or back to the planner."""
    opts = _opts(config)
    v = state.get("validation", {})
    if v.get("ok"):
        return "finalize"
    if state.get("attempts", 0) >= opts["max_attempts"]:
        return "finalize"
    if v.get("route") == "planner" and state.get("replans", 0) < opts["max_replans"]:
        return "planner"
    return "generator"


# --------------------------------------------------------------------------- #
async def finalize(state: AgentState, config: RunnableConfig) -> dict:
    _tools, llm, opts = _deps(config)
    ok = state.get("validation", {}).get("ok", False)
    out: dict = {"status": "success" if ok else "failed"}
    if not ok and state.get("last_good"):
        # Best effort: return the last query that at least executed
        out["sql"] = state["last_good"]["sql"]
        out["result"] = state["last_good"]["result"]
    result = out.get("result") or state.get("result")
    if opts["summarize"] and llm is not None and result and not result.get("error"):
        out["answer"] = (await llm.complete(
            P.SUMMARY_SYSTEM,
            P.SUMMARY_USER.format(question=state["question"], columns=result["columns"],
                                  row_count=result["row_count"],
                                  truncated=", truncated" if result.get("truncated") else "",
                                  preview=_preview(result, 30)),
            agent="summarizer",
        )).strip()
    title = "Done" if ok else f"Stopped after {state.get('attempts', 0)} attempts"
    out["trace"] = [event("system", title, status=out["status"])]
    return out
