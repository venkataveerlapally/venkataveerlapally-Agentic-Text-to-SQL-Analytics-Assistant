import sqlite3

import pytest

from config import settings
from mcp_server.db import SQLiteBackend, UnsafeSQLError, guard_read_only
from warehouse.catalog import CATALOG
from warehouse.seed import generate


def test_twelve_tables_with_data():
    assert len(CATALOG) == 12
    conn = sqlite3.connect(settings.sqlite_path)
    for t in CATALOG:
        assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] > 0, t


def test_generation_is_deterministic():
    a, b = generate(42), generate(42)
    assert a["orders"][:50] == b["orders"][:50]
    assert len(a["order_items"]) == len(b["order_items"])


def test_foreign_keys_hold():
    conn = sqlite3.connect(settings.sqlite_path)
    for table, meta in CATALOG.items():
        for col, _t, _d, fk in meta["columns"]:
            if not fk:
                continue
            ref_t, ref_c = fk.split(".")
            orphans = conn.execute(
                f"SELECT COUNT(*) FROM {table} x WHERE x.{col} IS NOT NULL AND NOT EXISTS "
                f"(SELECT 1 FROM {ref_t} y WHERE y.{ref_c} = x.{col})").fetchone()[0]
            assert orphans == 0, f"{table}.{col} -> {fk}"


@pytest.mark.parametrize("sql", [
    "SELECT 1",
    "select * from orders",
    "WITH x AS (SELECT 1 AS a) SELECT a FROM x",
    "```sql\nSELECT COUNT(*) FROM orders;\n```",
    "SELECT 'drop table orders' AS s",                 # keyword inside a string literal
    "SELECT REPLACE(email, '@', ' at ') FROM customers",  # REPLACE() function is fine
    "-- delete everything\nSELECT 1",
])
def test_guard_allows_reads(sql):
    guard_read_only(sql)


@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "DROP TABLE orders",
    "UPDATE orders SET status = 'x'",
    "WITH x AS (SELECT 1) DELETE FROM orders",
    "SELECT 1; DROP TABLE orders",
    "INSERT INTO regions VALUES (9, 'x')",
    "PRAGMA table_info(orders)",
    "ATTACH DATABASE 'x.db' AS x",
    "",
])
def test_guard_blocks_writes(sql):
    with pytest.raises(UnsafeSQLError):
        guard_read_only(sql)


def test_backend_is_read_only_even_without_guard():
    db = SQLiteBackend()
    conn = db._connect()
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM regions")


def test_backend_reports_errors_not_exceptions():
    db = SQLiteBackend()
    assert db.run_query("SELECT nope FROM orders")["error"]
    assert db.validate_sql("SELECT nope FROM orders")["valid"] is False
    assert db.validate_sql("SELECT order_id FROM orders")["valid"] is True
    r = db.run_query("SELECT * FROM order_items", limit=3)
    assert r["row_count"] == 3 and r["truncated"] is True
    with pytest.raises(ValueError):
        db.describe_table("users")
    with pytest.raises(ValueError):
        db.column_values("orders", "status; DROP TABLE orders")
