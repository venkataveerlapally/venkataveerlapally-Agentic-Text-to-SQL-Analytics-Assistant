"""Evaluate execution accuracy: single-shot baseline vs. the multi-agent system.

    python -m eval.run_eval                       # all questions, all modes
    python -m eval.run_eval --modes baseline,agent --limit 10
    python -m eval.run_eval --ids m02,h05 --verbose

Modes
  baseline     one LLM call, full schema, no planning or validation
  no_semantic  planner + generator + compile/runtime self-correction (no LLM validator)
  agent        full system: planner + generator + validator with semantic self-correction

Gold answers are always computed on the local SQLite warehouse (same data as
Snowflake), so the agent can run against either backend. Gold SQL is SQLite-dialect.
Results are written to eval/results/.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from eval.compare import results_match
from mcp_server.db import SQLiteBackend

HERE = Path(__file__).resolve().parent
MODES = ["baseline", "no_semantic", "agent"]


def load_questions(ids: list[str] | None = None, limit: int | None = None) -> list[dict]:
    qs = json.loads((HERE / "questions.json").read_text())
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    return qs[:limit] if limit else qs


def gold_results(questions: list[dict]) -> dict[str, list]:
    db = SQLiteBackend()
    out = {}
    for q in questions:
        r = db.run_query(q["sql"], limit=100_000)
        if r["error"]:
            raise RuntimeError(f"Gold SQL for {q['id']} failed: {r['error']}")
        out[q["id"]] = r["rows"]
    return out


async def evaluate(questions: list[dict], modes: list[str], verbose: bool = False) -> list[dict]:
    from agent.baseline import full_schema, single_shot
    from agent.graph import run_question
    from agent.llm import LLM
    from agent.mcp_client import WarehouseTools

    gold = gold_results(questions)
    records: list[dict] = []
    async with WarehouseTools() as tools:
        schema = await full_schema(tools)  # cached for the baseline
        for mode in modes:
            llm = LLM()
            print(f"\n=== {mode} ===")
            for i, q in enumerate(questions, 1):
                t0 = time.perf_counter()
                try:
                    if mode == "baseline":
                        out = await single_shot(q["question"], tools, llm, schema=schema)
                    else:
                        out = await run_question(q["question"], tools=tools, llm=llm,
                                                 options={"semantic_check": mode == "agent"})
                    res = out.get("result") or {}
                    err = res.get("error")
                    correct = (not err) and bool(res.get("columns")) and results_match(
                        gold[q["id"]], res.get("rows", []), q.get("order_matters", False))
                except Exception as e:  # noqa: BLE001 - one bad question shouldn't kill the run
                    out, err, correct = {}, f"{type(e).__name__}: {e}", False
                rec = {
                    "mode": mode, "id": q["id"], "difficulty": q["difficulty"], "question": q["question"],
                    "correct": correct, "status": out.get("status"), "attempts": out.get("attempts", 0),
                    "llm_calls": out.get("llm_calls", 0), "seconds": round(time.perf_counter() - t0, 2),
                    "sql": out.get("sql"), "error": err,
                }
                records.append(rec)
                mark = "PASS" if correct else "FAIL"
                print(f"[{i:>2}/{len(questions)}] {mark} {q['id']} attempts={rec['attempts']} "
                      f"calls={rec['llm_calls']} {rec['seconds']}s" + (f"  ({err[:80]})" if err else ""))
                if verbose and not correct:
                    print(f"      Q: {q['question']}\n      SQL: {rec['sql']}")
            print(f"tokens: in={llm.usage.input_tokens:,} out={llm.usage.output_tokens:,}")
    return records


def summarize(records: list[dict], modes: list[str]) -> str:
    lines = []
    diffs = ["easy", "medium", "hard"]
    header = f"{'mode':<12} {'overall':>9} " + " ".join(f"{d:>8}" for d in diffs) + f" {'avg calls':>10} {'avg s':>7}"
    lines += [header, "-" * len(header)]
    acc = {}
    for m in modes:
        rs = [r for r in records if r["mode"] == m]
        if not rs:
            continue
        by = defaultdict(list)
        for r in rs:
            by[r["difficulty"]].append(r["correct"])
        overall = sum(r["correct"] for r in rs) / len(rs)
        acc[m] = overall
        cells = " ".join(f"{(sum(by[d]) / len(by[d]) if by[d] else 0):>8.0%}" for d in diffs)
        lines.append(f"{m:<12} {overall:>9.1%} {cells} {sum(r['llm_calls'] for r in rs) / len(rs):>10.1f} "
                     f"{sum(r['seconds'] for r in rs) / len(rs):>7.1f}")

    agent_rs = [r for r in records if r["mode"] == "agent"]
    if agent_rs:
        retried = [r for r in agent_rs if r["attempts"] > 1]
        lines.append("")
        lines.append(f"Self-correction: {len(retried)}/{len(agent_rs)} questions needed >1 attempt; "
                     f"{sum(r['correct'] for r in retried)} of those ended correct.")
    if "baseline" in acc and "agent" in acc:
        b, a = acc["baseline"], acc["agent"]
        err_cut = (a - b) / (1 - b) if b < 1 else 0
        lines.append(f"Accuracy: {b:.0%} (baseline) -> {a:.0%} (agent), "
                     f"+{(a - b) * 100:.0f} pts, {err_cut:.0%} of baseline errors eliminated.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--ids", default="")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = set(modes) - set(MODES)
    if bad:
        raise SystemExit(f"Unknown mode(s): {bad}. Choose from {MODES}")

    questions = load_questions([i.strip() for i in args.ids.split(",") if i.strip()] or None, args.limit)
    records = asyncio.run(evaluate(questions, modes, args.verbose))
    report = summarize(records, modes)
    print("\n" + report)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (out_dir / f"eval-{stamp}.json").write_text(json.dumps({"summary": report, "records": records}, indent=2))
    with open(out_dir / f"eval-{stamp}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"\nSaved eval/results/eval-{stamp}.json and .csv")


if __name__ == "__main__":
    main()
