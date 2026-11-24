"""MCP server exposing the warehouse to the agents.

Tools: get_dialect, list_tables, describe_table, sample_rows, column_values,
validate_sql, run_query. Every query is read-only (see db.guard_read_only).

Run standalone (stdio transport):
    python -m mcp_server.server

Because it's a standard MCP server, you can also plug it into any MCP client
(e.g. Claude Desktop) to browse the warehouse directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # allow `python mcp_server/server.py`

try:  # MCP Python SDK v2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # MCP Python SDK v1
    from mcp.server.fastmcp import FastMCP as _Server

from mcp_server.db import get_backend

mcp = _Server("warehouse")
_backend = None


def backend():
    global _backend
    if _backend is None:
        _backend = get_backend()
    return _backend


@mcp.tool()
def get_dialect() -> dict:
    """Return the SQL dialect of the warehouse ('sqlite' or 'snowflake')."""
    return {"dialect": backend().dialect}


@mcp.tool()
def list_tables() -> dict:
    """List all warehouse tables with a business description and row count."""
    return {"tables": backend().list_tables()}


@mcp.tool()
def describe_table(table: str) -> dict:
    """Describe a table: columns, types, descriptions and foreign-key references."""
    try:
        return backend().describe_table(table)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def sample_rows(table: str, n: int = 3) -> dict:
    """Return up to n (max 10) example rows from a table."""
    try:
        return backend().sample_rows(table, n)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def column_values(table: str, column: str, limit: int = 25) -> dict:
    """Return the most common distinct values of a column with their counts.
    Useful for grounding filters like status = 'Cancelled'."""
    try:
        return backend().column_values(table, column, limit)
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def validate_sql(sql: str) -> dict:
    """Check that a query is read-only and compiles (via EXPLAIN) without running it."""
    return backend().validate_sql(sql)


@mcp.tool()
def run_query(sql: str, limit: int = 200) -> dict:
    """Run a read-only SELECT query and return columns and up to `limit` rows."""
    return backend().run_query(sql, limit)


if __name__ == "__main__":
    mcp.run()
