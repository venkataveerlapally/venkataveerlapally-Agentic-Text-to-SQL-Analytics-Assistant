"""Central configuration, read from environment variables / .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent

try:  # optional dependency
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    # "sqlite" (default, zero setup) or "snowflake"
    backend: str = field(default_factory=lambda: _env("WAREHOUSE_BACKEND", "sqlite").lower())
    sqlite_path: str = field(default_factory=lambda: _env("SQLITE_PATH", str(ROOT / "data" / "warehouse.db")))

    # LLM: "anthropic" (Claude API), "groq" / "gemini" (free cloud tiers) or "ollama" (free, local)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "anthropic").lower())
    groq_model: str = field(default_factory=lambda: _env("GROQ_MODEL", "openai/gpt-oss-120b"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-flash"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5-coder:7b"))
    ollama_base_url: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_num_ctx: int = field(default_factory=lambda: int(_env("OLLAMA_NUM_CTX", "8192")))
    ollama_timeout_s: int = field(default_factory=lambda: int(_env("OLLAMA_TIMEOUT_S", "300")))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-sonnet-5-5"))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "2048")))

    # Agent behaviour
    max_attempts: int = field(default_factory=lambda: int(_env("AGENT_MAX_ATTEMPTS", "4")))
    preview_rows: int = field(default_factory=lambda: int(_env("AGENT_PREVIEW_ROWS", "20")))
    result_row_limit: int = field(default_factory=lambda: int(_env("RESULT_ROW_LIMIT", "1000")))
    query_timeout_s: int = field(default_factory=lambda: int(_env("QUERY_TIMEOUT_S", "30")))

    # Snowflake
    snowflake_account: str = field(default_factory=lambda: _env("SNOWFLAKE_ACCOUNT"))
    snowflake_user: str = field(default_factory=lambda: _env("SNOWFLAKE_USER"))
    snowflake_password: str = field(default_factory=lambda: _env("SNOWFLAKE_PASSWORD"))
    snowflake_warehouse: str = field(default_factory=lambda: _env("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"))
    snowflake_database: str = field(default_factory=lambda: _env("SNOWFLAKE_DATABASE", "TEXT2SQL"))
    snowflake_schema: str = field(default_factory=lambda: _env("SNOWFLAKE_SCHEMA", "RETAIL"))
    snowflake_role: str = field(default_factory=lambda: _env("SNOWFLAKE_ROLE"))  # read-only role for the agent
    snowflake_admin_role: str = field(default_factory=lambda: _env("SNOWFLAKE_ADMIN_ROLE"))  # for loading

    def snowflake_params(self, admin: bool = False) -> dict:
        params = {
            "account": self.snowflake_account,
            "user": self.snowflake_user,
            "password": self.snowflake_password,
            "warehouse": self.snowflake_warehouse,
            "database": self.snowflake_database,
            "schema": self.snowflake_schema,
        }
        role = self.snowflake_admin_role if admin else self.snowflake_role
        if role:
            params["role"] = role
        missing = [k for k in ("account", "user", "password") if not params[k]]
        if missing:
            raise RuntimeError(f"Missing Snowflake settings: {', '.join('SNOWFLAKE_' + m.upper() for m in missing)}")
        return params


settings = Settings()
