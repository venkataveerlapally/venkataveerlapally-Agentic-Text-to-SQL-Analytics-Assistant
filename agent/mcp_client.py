"""Async client for the warehouse MCP server (spawned as a stdio subprocess)."""

from __future__ import annotations

import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import ROOT


class MCPToolError(RuntimeError):
    pass


class WarehouseTools:
    """`async with WarehouseTools() as tools: await tools.call("list_tables")`

    Also logs every tool call so the UI can show exactly what the agents did.
    """

    def __init__(self, env: dict[str, str] | None = None):
        self._env = {**os.environ, **(env or {})}
        self._stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None
        self.call_log: list[dict] = []

    async def __aenter__(self) -> "WarehouseTools":
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            cwd=str(ROOT),
            env=self._env,
        )
        self._stack = AsyncExitStack()
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._stack:
            await self._stack.aclose()

    async def list_tool_names(self) -> list[str]:
        res = await self.session.list_tools()
        return [t.name for t in res.tools]

    async def call(self, tool: str, **args: Any) -> dict:
        res = await self.session.call_tool(tool, args)
        is_error = getattr(res, "is_error", None)
        if is_error is None:
            is_error = getattr(res, "isError", False)
        text = "".join(getattr(c, "text", "") for c in (res.content or []))
        if is_error:
            self.call_log.append({"tool": tool, "args": args, "error": text})
            raise MCPToolError(f"{tool} failed: {text}")
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            data = {"text": text}
        self.call_log.append({"tool": tool, "args": args})
        return data
