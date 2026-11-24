import pytest

from eval.compare import results_match
from eval.run_eval import gold_results, load_questions, summarize
from mcp_server.db import SQLiteBackend


def test_column_order_and_names_ignored():
    gold = [("East", 10.0), ("West", 20.0)]
    pred = [(20.0, "West"), (10.0, "East")]
    assert results_match(gold, pred)


def test_extra_columns_allowed_but_missing_not():
    gold = [("Electronics", 100.0)]
    assert results_match(gold, [(1, "Electronics", 100.0)])
    assert not results_match([("Electronics", 100.0, 5)], [("Electronics", 100.0)])


def test_rounding_tolerance():
    assert results_match([(392.2546951,)], [(392.25,)])
    assert results_match([(1997264.2364999,)], [(1997264.24,)])
    assert not results_match([(392.25,)], [(393.25,)])
    assert not results_match([(1997264.24,)], [(1990000.0,)])


def test_order_matters_flag():
    gold = [("a", 3.0), ("b", 2.0)]
    rev = [("b", 2.0), ("a", 3.0)]
    assert results_match(gold, rev, order_matters=False)
    assert not results_match(gold, rev, order_matters=True)


def test_numeric_strings_and_nulls():
    assert results_match([("2024", 5)], [(2024, 5.0)])
    assert results_match([(None,)], [(None,)])
    assert not results_match([(None,)], [(0,)])
    assert not results_match([(1,)], [(1,), (1,)])


def test_duplicate_column_values_need_consistent_mapping():
    gold = [(1.0, 2.0), (2.0, 1.0)]
    assert results_match(gold, [(2.0, 1.0), (1.0, 2.0)])
    assert not results_match([(1.0, 1.0), (2.0, 2.0)], [(1.0, 2.0), (2.0, 1.0)])


def test_question_set_is_valid():
    qs = load_questions()
    assert len(qs) == 40
    assert len({q["id"] for q in qs}) == len(qs)
    assert {q["difficulty"] for q in qs} == {"easy", "medium", "hard"}
    gold = gold_results(qs)
    for q in qs:
        assert gold[q["id"]], f"{q['id']} returns no rows"
        assert results_match(gold[q["id"]], gold[q["id"]], q.get("order_matters", False))


def test_known_wrong_answers_are_rejected():
    """Typical agent mistakes must score as wrong."""
    db = SQLiteBackend()
    gold = gold_results(load_questions(ids=["e06", "h05"]))
    # forgot to exclude cancelled orders
    wrong = db.run_query("SELECT SUM(oi.quantity*oi.unit_price*(1-oi.discount)) FROM order_items oi "
                         "JOIN orders o USING(order_id) WHERE o.order_date LIKE '2024%'")["rows"]
    assert not results_match(gold["e06"], wrong)
    # fan-out: joined items and payments before aggregating
    fanout = db.run_query(
        "SELECT SUM(oi.quantity*oi.unit_price*(1-oi.discount)), SUM(p.amount) FROM orders o "
        "JOIN order_items oi USING(order_id) JOIN payments p USING(order_id) "
        "WHERE o.status <> 'Cancelled' AND o.order_date LIKE '2025%'")["rows"]
    assert not results_match(gold["h05"], fanout)
    # an equivalent but differently written query is accepted
    alt = db.run_query("SELECT ROUND(SUM(quantity*unit_price*(1-discount)), 2) AS revenue FROM order_items "
                       "WHERE order_id IN (SELECT order_id FROM orders WHERE status != 'Cancelled' "
                       "AND strftime('%Y', order_date) = '2024')")["rows"]
    assert results_match(gold["e06"], alt)


def test_summary_report():
    recs = [
        {"mode": "baseline", "difficulty": "easy", "correct": True, "llm_calls": 1, "seconds": 1, "attempts": 1},
        {"mode": "baseline", "difficulty": "hard", "correct": False, "llm_calls": 1, "seconds": 1, "attempts": 1},
        {"mode": "agent", "difficulty": "easy", "correct": True, "llm_calls": 4, "seconds": 3, "attempts": 1},
        {"mode": "agent", "difficulty": "hard", "correct": True, "llm_calls": 6, "seconds": 5, "attempts": 2},
    ]
    report = summarize(recs, ["baseline", "agent"])
    assert "50% (baseline) -> 100% (agent)" in report
    assert "1/2 questions needed >1 attempt" in report
