"""Shared LangGraph state."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    dialect: str
    data_start: str
    data_end: str
    catalog: str                     # compact list of all tables (for the planner)

    plan: dict                       # planner output
    schema_context: str              # detailed schema for the planned tables
    replans: int                     # how many times the planner has been re-run

    sql: str                         # current candidate query
    attempts: int                    # generator attempts so far
    validation: dict                 # {"ok", "stage", "issues", "route"}
    result: dict                     # result of the current candidate
    last_good: dict                  # last query that executed without error {"sql", "result"}

    status: str                      # "success" | "failed"
    answer: str                      # optional natural-language summary

    # append-only logs (LangGraph merges these with operator.add)
    history: Annotated[list[dict], operator.add]   # failed attempts + feedback
    trace: Annotated[list[dict], operator.add]     # events for the UI


def event(agent: str, title: str, **data: Any) -> dict:
    return {"agent": agent, "title": title, **data}
