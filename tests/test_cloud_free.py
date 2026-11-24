"""OpenAICompatLLM (Groq / Gemini) against a fake OpenAI-style server."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.llm import OpenAICompatLLM
from tests.test_agent_loop import CORRECT, PLAN, drive


def start_fake_api(replies, requests, rate_limit_first=0, status=200):
    state = {"limited": rate_limit_first}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
            if state["limited"] > 0:
                state["limited"] -= 1
                self.send_response(429)
                self.send_header("retry-after", "1")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"rate limit"}}')
                return
            if status != 200:
                self.send_response(status)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"bad"}}')
                return
            system = body["messages"][0]["content"]
            agent = "planner" if "PLANNER" in system else "validator" if "VALIDATOR" in system else "generator"
            out = {"choices": [{"message": {"role": "assistant", "content": replies[agent].pop(0)}}],
                   "usage": {"prompt_tokens": 50, "completion_tokens": 10}}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/openai/v1"


def test_agent_runs_on_groq_with_rate_limit_retry():
    requests = []
    server, url = start_fake_api({
        "planner": [json.dumps(PLAN)],
        "generator": [f"```sql\n{CORRECT}\n```"],
        "validator": ['{"verdict": "pass", "issues": [], "suggestion": ""}'],
    }, requests, rate_limit_first=1)
    try:
        llm = OpenAICompatLLM("groq", model="openai/gpt-oss-120b", base_url=url, api_key="gsk_test")
        s = asyncio.run(drive("What was total revenue in 2024?", llm))
    finally:
        server.shutdown()
    assert s["status"] == "success"
    assert llm.usage.calls == 3 and llm.usage.input_tokens == 150
    assert len(requests) == 4  # one 429 + three successful calls
    assert requests[0]["path"] == "/openai/v1/chat/completions"
    assert requests[0]["auth"] == "Bearer gsk_test"
    assert requests[1]["body"]["reasoning_effort"] == "low"


def test_bad_key_and_missing_key(monkeypatch):
    server, url = start_fake_api({}, [], status=401)
    try:
        with pytest.raises(RuntimeError, match="rejected the API key"):
            asyncio.run(OpenAICompatLLM("groq", base_url=url, api_key="x").complete("s", "u"))
    finally:
        server.shutdown()
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="aistudio.google.com"):
        OpenAICompatLLM("gemini")


def test_factory_picks_groq(monkeypatch):
    from agent import llm as llm_mod
    from config import settings

    monkeypatch.setattr(settings, "llm_provider", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_x")
    client = llm_mod.LLM()
    assert isinstance(client, OpenAICompatLLM) and client.model == settings.groq_model
