"""OllamaLLM against a fake Ollama HTTP server (no real model needed)."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agent.llm import OllamaLLM
from config import settings
from tests.test_agent_loop import CORRECT, PLAN, drive


def start_fake_ollama(replies: dict[str, list[str]], requests: list[dict], missing_model: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            if missing_model:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'{"error":"model \\"x\\" not found, try pulling it first"}')
                return
            system = body["messages"][0]["content"]
            agent = "planner" if "PLANNER" in system else "validator" if "VALIDATOR" in system else "generator"
            out = {"message": {"role": "assistant", "content": replies[agent].pop(0)},
                   "prompt_eval_count": 100, "eval_count": 20, "done": True}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def test_agent_runs_on_ollama():
    requests: list[dict] = []
    server, url = start_fake_ollama({
        "planner": [json.dumps(PLAN)],
        "generator": ["Here you go:\n```sql\nSELECT nope FROM orders\n```", f"```sql\n{CORRECT}\n```"],
        "validator": ['{"verdict": "pass", "issues": [], "suggestion": ""}'],
    }, requests)
    try:
        llm = OllamaLLM(model="qwen2.5-coder:7b", base_url=url)
        s = asyncio.run(drive("What was total revenue in 2024?", llm))
    finally:
        server.shutdown()

    assert s["status"] == "success" and s["attempts"] == 2
    assert llm.usage.calls == 4 and llm.usage.input_tokens == 400
    assert requests[0]["model"] == "qwen2.5-coder:7b"
    assert requests[0]["stream"] is False
    assert requests[0]["options"]["num_ctx"] == settings.ollama_num_ctx
    assert requests[0]["format"] == "json"      # planner uses JSON mode
    assert "format" not in requests[1]          # generator returns SQL text


def test_helpful_errors():
    server, url = start_fake_ollama({}, [], missing_model=True)
    try:
        with pytest.raises(RuntimeError, match="ollama pull"):
            asyncio.run(OllamaLLM(model="x", base_url=url).complete("s", "u"))
    finally:
        server.shutdown()
    with pytest.raises(RuntimeError, match="Is the Ollama app running"):
        asyncio.run(OllamaLLM(model="x", base_url="http://127.0.0.1:9").complete("s", "u"))


def test_factory_picks_provider(monkeypatch):
    from agent import llm as llm_mod

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    assert isinstance(llm_mod.LLM(), OllamaLLM)
    monkeypatch.setattr(settings, "llm_provider", "nope")
    with pytest.raises(ValueError):
        llm_mod.LLM()
