"""In-process fake of an OpenAI-compatible /v1/chat/completions endpoint, for tests without GPUs or network.

Two ways to use it:
1. `httpx.MockTransport` (preferred, no sockets):  transport = make_transport(responder)
   and pass it to the client under test (LLMClient(..., transport=transport)).
2. A real local HTTP server in a thread (for CLI subprocess tests): with FakeServer(responder) as srv: srv.base_url

`responder(request_json: dict, call_index: int) -> dict` returns the JSON body of a chat completion. Helpers build
plain-text responses, tool-call responses, and reasoning-bearing responses.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import httpx

Responder = Callable[[dict, int], dict]


def text_response(content: str, *, model: str = "fake", reasoning: str | None = None,
                  finish_reason: str = "stop", prompt_tokens: int = 10, completion_tokens: int = 5) -> dict:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                  "total_tokens": prompt_tokens + completion_tokens},
    }


def tool_call_response(calls: list[tuple[str, dict]], *, content: str | None = None, model: str = "fake",
                       reasoning: str | None = None) -> dict:
    """calls = [(function_name, arguments_dict), ...]"""
    msg: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": [
        {"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}} for name, args in calls]}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
        "model": model, "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class ScriptedResponder:
    """Returns the scripted responses in order; records every request. Extra calls repeat the last response."""

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, request_json: dict, call_index: int) -> dict:
        with self._lock:
            self.requests.append(request_json)
            i = min(len(self.requests) - 1, len(self.responses) - 1)
            return self.responses[i]


def make_transport(responder: Responder) -> httpx.MockTransport:
    counter = {"n": 0}
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"object": "list", "data": [{"id": "fake", "object": "model"}]})
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content or b"{}")
            with lock:
                idx = counter["n"]
                counter["n"] += 1
            out = responder(body, idx)
            if isinstance(out, tuple):  # (status_code, json) to simulate errors
                return httpx.Response(out[0], json=out[1])
            return httpx.Response(200, json=out)
        return httpx.Response(404, json={"error": "not found"})

    return httpx.MockTransport(handler)


class FakeServer:
    """Real localhost HTTP server (thread) speaking the minimal OpenAI chat API. Use as a context manager."""

    def __init__(self, responder: Responder, host: str = "127.0.0.1", port: int = 0):
        self.responder = responder
        self.counter = 0
        self.lock = threading.Lock()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.endswith("/chat/completions"):
                    self._send(404, {"error": "not found"}); return
                with server.lock:
                    idx = server.counter; server.counter += 1
                out = server.responder(body, idx)
                if isinstance(out, tuple):
                    self._send(out[0], out[1])
                else:
                    self._send(200, out)

            def _send(self, code: int, obj: dict):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        h, p = self.httpd.server_address[:2]
        return f"http://{h}:{p}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
