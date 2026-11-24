"""Warehouse backends (SQLite / Snowflake) behind one small interface.

Every query entry point goes through `guard_read_only` first, so the MCP
server can only ever run a single SELECT/WITH statement, whatever the LLM
writes. For Snowflake, also connect with a read-only role (defense in depth).
"""

from __future__ import annotations

import datetime as dt
import decimal
import re
import sqlite3
import time
from typing import Any

from config import settings
from warehouse.catalog import CATALOG

# --------------------------------------------------------------------------- #
# Read-only SQL guard
# --------------------------------------------------------------------------- #
_FORBIDDEN = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bMERGE\b", r"\bDROP\b", r"\bCREATE\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bGRANT\b", r"\bREVOKE\b", r"\bREPLACE\s+INTO\b",
    r"\bCOPY\s+INTO\b", r"\bPUT\s+file:", r"\bGET\s+@", r"\bCALL\b", r"\bATTACH\b", r"\bDETACH\b",
    r"\bPRAGMA\b", r"\bVACUUM\b", r"\bEXECUTE\b", r"\bUNDROP\b", r"\bUSE\s+(ROLE|WAREHOUSE|DATABASE|SCHEMA)\b",
]


class UnsafeSQLError(ValueError):
    pass


def strip_sql(sql: str) -> str:
    """Remove comments and string-literal contents (for keyword scanning only)."""
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    sql = re.sub(r"'(?:[^']|'')*'", "''", sql)
    return sql


def clean_sql(sql: str) -> str:
    """Normalise LLM output: drop markdown fences and trailing semicolons."""
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.I)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";").strip()


def guard_read_only(sql: str) -> str:
    sql = clean_sql(sql)
    if not sql:
        raise UnsafeSQLError("Empty SQL.")
    scan = strip_sql(sql)
    if ";" in scan:
        raise UnsafeSQLError("Only a single SQL statement is allowed.")
    first = re.match(r"\s*\(?\s*(\w+)", scan)
    if not first or first.group(1).upper() not in ("SELECT", "WITH"):
        raise UnsafeSQLError("Only read-only SELECT / WITH queries are allowed.")
    for pat in _FORBIDDEN:
        m = re.search(pat, scan, flags=re.I)
        if m:
            raise UnsafeSQLError(f"Forbidden keyword in query: {m.group(0).strip().upper()}")
    return sql


def _jsonable(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (dt.date, dt.datetime, dt.time)):
        return v.isoformat()
    if isinstance(v, bytes):
        return v.hex()
    return v


def _check_table(table: str) -> str:
    t = table.strip().lower()
    if t not in CATALOG:
        raise ValueError(f"Unknown table '{table}'. Known tables: {', '.join(CATALOG)}")
    return t


def _check_column(table: str, column: str) -> str:
    c = column.strip().lower()
    if c not in [col[0] for col in CATALOG[table]["columns"]]:
        raise ValueError(f"Unknown column '{column}' in table '{table}'.")
    return c


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
class Backend:
    dialect: str = ""

    # --- subclass hooks --------------------------------------------------- #
    def _execute(self, sql: str, limit: int) -> tuple[list[str], list[list[Any]], bool]:
        raise NotImplementedError

    def _explain(self, sql: str) -> None:
        raise NotImplementedError

    def _row_counts(self) -> dict[str, int]:
        raise NotImplementedError

    # --- public API used by MCP tools ------------------------------------- #
    def list_tables(self) -> list[dict]:
        counts = self._row_counts()
        return [
            {"table": t, "description": meta["description"], "row_count": counts.get(t)}
            for t, meta in CATALOG.items()
        ]

    def describe_table(self, table: str) -> dict:
        t = _check_table(table)
        meta = CATALOG[t]
        return {
            "table": t,
            "description": meta["description"],
            "columns": [
                {"name": n, "type": typ, "description": d, **({"references": fk} if fk else {})}
                for n, typ, d, fk in meta["columns"]
            ],
        }

    def sample_rows(self, table: str, n: int = 3) -> dict:
        t = _check_table(table)
        n = max(1, min(int(n), 10))
        return self.run_query(f"SELECT * FROM {t} LIMIT {n}", limit=n)

    def column_values(self, table: str, column: str, limit: int = 25) -> dict:
        t = _check_table(table)
        c = _check_column(t, column)
        limit = max(1, min(int(limit), 100))
        return self.run_query(
            f"SELECT {c} AS value, COUNT(*) AS n FROM {t} GROUP BY {c} ORDER BY n DESC LIMIT {limit}",
            limit=limit,
        )

    def validate_sql(self, sql: str) -> dict:
        try:
            sql = guard_read_only(sql)
            self._explain(sql)
            return {"valid": True, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"valid": False, "error": f"{type(e).__name__}: {e}"}

    def run_query(self, sql: str, limit: int | None = None) -> dict:
        limit = int(limit or settings.result_row_limit)
        start = time.perf_counter()
        try:
            sql = guard_read_only(sql)
            cols, rows, truncated = self._execute(sql, limit)
            return {
                "columns": cols,
                "rows": [[_jsonable(v) for v in r] for r in rows],
                "row_count": len(rows),
                "truncated": truncated,
                "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                "error": None,
            }
        except Exception as e:  # noqa: BLE001
            return {"columns": [], "rows": [], "row_count": 0, "truncated": False,
                    "elapsed_ms": round((time.perf_counter() - start) * 1000, 1),
                    "error": f"{type(e).__name__}: {e}"}


class SQLiteBackend(Backend):
    dialect = "sqlite"

    def __init__(self, path: str | None = None):
        self.path = path or settings.sqlite_path

    def _connect(self) -> sqlite3.Connection:
        # mode=ro: the file is opened read-only at the OS level
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=5)
        conn.execute("PRAGMA query_only = ON")
        deadline = time.monotonic() + settings.query_timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        return conn

    def _execute(self, sql, limit):
        conn = self._connect()
        try:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(limit + 1)
            return cols, [list(r) for r in rows[:limit]], len(rows) > limit
        finally:
            conn.close()

    def _explain(self, sql):
        conn = self._connect()
        try:
            conn.execute(f"EXPLAIN QUERY PLAN {sql}").fetchall()
        finally:
            conn.close()

    def _row_counts(self):
        conn = self._connect()
        try:
            return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in CATALOG}
        finally:
            conn.close()


class SnowflakeBackend(Backend):
    dialect = "snowflake"

    def __init__(self):
        import snowflake.connector  # imported lazily so SQLite users don't need it

        self._conn = snowflake.connector.connect(**settings.snowflake_params(admin=False))

    def _execute(self, sql, limit):
        cur = self._conn.cursor()
        try:
            cur.execute(sql, timeout=settings.query_timeout_s)
            cols = [d[0].lower() for d in cur.description or []]
            rows = cur.fetchmany(limit + 1)
            return cols, [list(r) for r in rows[:limit]], len(rows) > limit
        finally:
            cur.close()

    def _explain(self, sql):
        cur = self._conn.cursor()
        try:
            cur.execute(f"EXPLAIN USING TEXT {sql}", timeout=settings.query_timeout_s)
        finally:
            cur.close()

    def _row_counts(self):
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT LOWER(table_name), row_count FROM information_schema.tables "
                "WHERE table_schema = %s",
                (settings.snowflake_schema.upper(),),
            )
            return {name: int(n or 0) for name, n in cur.fetchall()}
        finally:
            cur.close()


def get_backend() -> Backend:
    if settings.backend == "snowflake":
        return SnowflakeBackend()
    if settings.backend == "sqlite":
        return SQLiteBackend()
    raise ValueError(f"Unknown WAREHOUSE_BACKEND '{settings.backend}' (use sqlite or snowflake)")
