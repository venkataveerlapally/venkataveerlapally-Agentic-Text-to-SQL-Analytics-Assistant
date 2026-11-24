"""LLM clients with usage accounting.

Two providers share one interface (complete / complete_json / usage / model):
  * anthropic : Claude via the Anthropic API (needs ANTHROPIC_API_KEY and credits)
  * ollama    : a free local model via Ollama (https://ollama.com), no API key
  * groq      : free cloud tier at console.groq.com (no credit card), OpenAI-compatible API
  * gemini    : Google AI Studio free tier, via its OpenAI-compatible endpoint

Pick with LLM_PROVIDER in .env. Call sites just use `LLM(...)`.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from config import settings


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_agent: dict[str, int] = field(default_factory=dict)


class AnthropicLLM:
    provider = "anthropic"

    def __init__(self, model: str | None = None, max_tokens: int | None = None):
        import anthropic  # lazy import so tooling/tests don't need the SDK

        self.client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        self.model = model or settings.anthropic_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.usage = Usage()
        self._use_temperature = True

    async def complete(self, system: str, user: str, *, agent: str = "llm") -> str:
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                      messages=[{"role": "user", "content": user}])
        if self._use_temperature:
            kwargs["temperature"] = 0.0
        try:
            resp = await self.client.messages.create(**kwargs)
        except Exception as e:  # some models reject sampling params; retry without
            if self._use_temperature and "temperature" in str(e).lower():
                self._use_temperature = False
                kwargs.pop("temperature", None)
                resp = await self.client.messages.create(**kwargs)
            else:
                raise
        self.usage.calls += 1
        self.usage.by_agent[agent] = self.usage.by_agent.get(agent, 0) + 1
        if getattr(resp, "usage", None):
            self.usage.input_tokens += resp.usage.input_tokens
            self.usage.output_tokens += resp.usage.output_tokens
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def complete_json(self, system: str, user: str, *, agent: str = "llm") -> dict:
        text = await self.complete(system, user, agent=agent)
        return parse_json(text)


class OllamaLLM:
    """Local model served by Ollama. Uses only the standard library (no extra packages)."""

    provider = "ollama"

    def __init__(self, model: str | None = None, max_tokens: int | None = None, base_url: str | None = None):
        self.model = model or settings.ollama_model
        self.max_tokens = max_tokens or settings.max_tokens
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.usage = Usage()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(f"{self.base_url}/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=settings.ollama_timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 404 and "not found" in body.lower():
                raise RuntimeError(f"Ollama model '{self.model}' is not downloaded. Run: ollama pull {self.model}") from e
            raise RuntimeError(f"Ollama error {e.code}: {body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Can't reach Ollama at {self.base_url}. Is the Ollama app running?") from e

    async def _chat(self, system: str, user: str, agent: str, json_mode: bool) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            # num_ctx: Ollama's default context is small and would silently cut off the schema
            "options": {"temperature": 0, "num_ctx": settings.ollama_num_ctx, "num_predict": self.max_tokens},
        }
        if json_mode:
            payload["format"] = "json"
        data = await asyncio.to_thread(self._post, payload)
        self.usage.calls += 1
        self.usage.by_agent[agent] = self.usage.by_agent.get(agent, 0) + 1
        self.usage.input_tokens += int(data.get("prompt_eval_count") or 0)
        self.usage.output_tokens += int(data.get("eval_count") or 0)
        return data.get("message", {}).get("content", "")

    async def complete(self, system: str, user: str, *, agent: str = "llm") -> str:
        return await self._chat(system, user, agent, json_mode=False)

    async def complete_json(self, system: str, user: str, *, agent: str = "llm") -> dict:
        return parse_json(await self._chat(system, user, agent, json_mode=True))


class OpenAICompatLLM:
    """Any OpenAI-compatible chat API (Groq, Gemini, ...). Standard library only.

    Free tiers have tight per-minute limits, so HTTP 429 responses are retried
    after the wait time the server asks for.
    """

    PRESETS = {
        "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
                 "signup": "https://console.groq.com/keys"},
        "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY",
                   "signup": "https://aistudio.google.com/apikey"},
    }

    def __init__(self, provider: str, model: str | None = None, max_tokens: int | None = None,
                 base_url: str | None = None, api_key: str | None = None, max_retries: int = 6):
        import os

        preset = self.PRESETS[provider]
        self.provider = provider
        self.model = model or (settings.groq_model if provider == "groq" else settings.gemini_model)
        self.max_tokens = max_tokens or settings.max_tokens
        self.base_url = (base_url or preset["base_url"]).rstrip("/")
        self.api_key = api_key or os.environ.get(preset["key_env"], "")
        if not self.api_key:
            raise RuntimeError(f"{preset['key_env']} is not set. Get a free key at {preset['signup']} "
                               f"and add {preset['key_env']}=... to your .env file.")
        self.max_retries = max_retries
        self.usage = Usage()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
                     "User-Agent": "text2sql-agent/1.0"},
        )
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="replace")
                if e.code in (429, 503) and attempt < self.max_retries:
                    wait = _retry_after(e.headers.get("retry-after"), default=min(60, 5 * 2 ** attempt))
                    print(f"[{self.provider}] rate limited, waiting {wait:.0f}s ...", flush=True)
                    time.sleep(wait)
                    continue
                if e.code in (401, 403):
                    raise RuntimeError(f"{self.provider} rejected the API key ({e.code}). Check it in .env. {body[:200]}") from e
                if e.code == 404:
                    raise RuntimeError(f"{self.provider} doesn't know model '{self.model}'. "
                                       f"Pick one from its models list and set it in .env. {body[:200]}") from e
                raise RuntimeError(f"{self.provider} error {e.code}: {body[:500]}") from e
            except urllib.error.URLError as e:
                raise RuntimeError(f"Can't reach {self.base_url}: {e.reason}") from e
        raise RuntimeError(f"{self.provider}: still rate limited after {self.max_retries} retries")

    async def complete(self, system: str, user: str, *, agent: str = "llm") -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"  # keeps token use under free-tier per-minute limits
        data = await asyncio.to_thread(self._post, payload)
        self.usage.calls += 1
        self.usage.by_agent[agent] = self.usage.by_agent.get(agent, 0) + 1
        u = data.get("usage") or {}
        self.usage.input_tokens += int(u.get("prompt_tokens") or 0)
        self.usage.output_tokens += int(u.get("completion_tokens") or 0)
        choices = data.get("choices") or [{}]
        return (choices[0].get("message") or {}).get("content") or ""

    async def complete_json(self, system: str, user: str, *, agent: str = "llm") -> dict:
        return parse_json(await self.complete(system, user, agent=agent))


def _retry_after(value: str | None, default: float) -> float:
    try:
        return max(1.0, min(120.0, float(value))) if value else default
    except ValueError:
        return default


def LLM(model: str | None = None, max_tokens: int | None = None):  # noqa: N802 - factory used like a class
    """Return the client for the configured provider (LLM_PROVIDER)."""
    if settings.llm_provider in OpenAICompatLLM.PRESETS:
        return OpenAICompatLLM(settings.llm_provider, model=model, max_tokens=max_tokens)
    if settings.llm_provider == "ollama":
        return OllamaLLM(model=model, max_tokens=max_tokens)
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(model=model, max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (use anthropic, groq, gemini or ollama)")


def parse_json(text: str) -> dict:
    """Extract the first JSON object from model output (tolerates ```json fences)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = text.find("{")
        if start == -1:
            raise ValueError(f"No JSON object in model output: {text[:200]!r}")
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    break
    if candidate is None:
        raise ValueError(f"Unbalanced JSON in model output: {text[:200]!r}")
    return json.loads(candidate)


def extract_sql(text: str) -> str:
    """Pull SQL out of a ```sql block, or fall back to the raw text."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = m.group(1) if m else text
    return sql.strip().rstrip(";").strip()
