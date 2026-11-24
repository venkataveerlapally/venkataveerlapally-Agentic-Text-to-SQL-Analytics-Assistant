"""Single-shot baseline: one LLM call with the full schema, no planning, no validation.

This is the comparison point for the evaluation harness. It gets the SAME business
rules and dialect notes as the agent, so any accuracy gain comes from the
multi-agent architecture (planning + grounding + self-correction), not from
withholding information from the baseline.
"""

from __future__ import annotations

from agent import prompts as P
from agent.llm import extract_sql
from agent.mcp_client import WarehouseTools
from config import settings


async def full_schema(tools: WarehouseTools) -> str:
    tables = (await tools.call("list_tables"))["tables"]
    parts = []
    for t in tables:
        d = await tools.call("describe_table", table=t["table"])
        cols = "\n".join(
            f"    {c['name']} {c['type']}  -- {c['description']}"
            + (f" (references {c['references']})" if c.get("references") else "")
            for c in d["columns"]
        )
        parts.append(f"TABLE {d['table']}: {d['description']}\n{cols}")
    return "\n\n".join(parts)


async def single_shot(question: str, tools: WarehouseTools, llm, *, schema: str | None = None,
                      dialect: str | None = None, data_range: tuple[str, str] | None = None) -> dict:
    dialect = dialect or (await tools.call("get_dialect"))["dialect"]
    schema = schema or await full_schema(tools)
    if data_range is None:
        r = await tools.call("run_query", sql="SELECT MIN(order_date), MAX(order_date) FROM orders", limit=1)
        data_range = tuple(str(x) for x in r["rows"][0])
    rules = P.COMMON_RULES.format(data_start=data_range[0], data_end=data_range[1])
    calls_before = llm.usage.calls
    text = await llm.complete(
        P.BASELINE_SYSTEM.format(dialect=dialect, rules=rules, dialect_notes=P.DIALECT_NOTES.get(dialect, "")),
        P.BASELINE_USER.format(schema=schema, question=question),
        agent="baseline",
    )
    sql = extract_sql(text)
    result = await tools.call("run_query", sql=sql, limit=settings.result_row_limit)
    return {
        "sql": sql,
        "result": result,
        "status": "failed" if result.get("error") else "success",
        "attempts": 1,
        "llm_calls": llm.usage.calls - calls_before,
    }
