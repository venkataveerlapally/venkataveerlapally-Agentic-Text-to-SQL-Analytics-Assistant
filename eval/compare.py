"""Execution-accuracy comparison of a predicted result set against a gold result set.

A prediction is correct when it returns the same rows as the gold query, where:
  * column names and column order are ignored
  * the prediction may include EXTRA columns (e.g. category_id next to category_name),
    as long as some subset of its columns reproduces the gold result exactly
  * row order only matters when the question's gold query is marked order_matters
  * numbers match within a tolerance that absorbs rounding to 2 decimals
  * numeric strings ('2025') and numbers (2025) are treated as equal
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")


def norm(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if _NUM_RE.match(s):
        return float(s)
    return s.lower()


def values_equal(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        # absolute 0.0051 covers rounding to 2 decimals; relative 1e-5 covers sums of rounded lines
        return abs(a - b) <= max(0.0051 + 1e-9, 1e-5 * max(abs(a), abs(b)))
    return a == b


def _sort_key(row: Sequence[Any]) -> tuple:
    # None < numbers < strings, so mixed columns still sort deterministically
    return tuple((0, 0) if v is None else (1, v) if isinstance(v, float) else (2, v) for v in row)


def _rows_equal(gold: list[tuple], pred: list[tuple], order_matters: bool) -> bool:
    if len(gold) != len(pred):
        return False
    if not order_matters:
        gold, pred = sorted(gold, key=_sort_key), sorted(pred, key=_sort_key)
    return all(all(values_equal(g, p) for g, p in zip(gr, pr)) for gr, pr in zip(gold, pred))


def _column_compatible(gold_col: list, pred_col: list) -> bool:
    g = sorted(gold_col, key=lambda v: _sort_key((v,)))
    p = sorted(pred_col, key=lambda v: _sort_key((v,)))
    return all(values_equal(a, b) for a, b in zip(g, p))


def results_match(gold_rows: list[Sequence[Any]], pred_rows: list[Sequence[Any]], order_matters: bool = False) -> bool:
    gold = [tuple(norm(v) for v in r) for r in gold_rows]
    pred = [tuple(norm(v) for v in r) for r in pred_rows]
    if len(gold) != len(pred):
        return False
    if not gold:
        return True
    n_gold, n_pred = len(gold[0]), len(pred[0])
    if n_pred < n_gold:
        return False

    gold_cols = [[r[j] for r in gold] for j in range(n_gold)]
    pred_cols = [[r[k] for r in pred] for k in range(n_pred)]
    candidates = [[k for k in range(n_pred) if _column_compatible(gold_cols[j], pred_cols[k])] for j in range(n_gold)]
    if any(not c for c in candidates):
        return False

    # Backtracking search over injective gold->pred column mappings
    order = sorted(range(n_gold), key=lambda j: len(candidates[j]))
    mapping: dict[int, int] = {}

    def search(i: int) -> bool:
        if i == len(order):
            projected = [tuple(r[mapping[j]] for j in range(n_gold)) for r in pred]
            return _rows_equal(gold, projected, order_matters)
        j = order[i]
        for k in candidates[j]:
            if k in mapping.values():
                continue
            mapping[j] = k
            if search(i + 1):
                return True
            del mapping[j]
        return False

    return search(0)
