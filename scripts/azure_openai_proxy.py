"""Tiny HTTP shim that adapts Azure's OpenAI-compatible v1 endpoint to
ensemble's openai backend.

Ensemble's openai backend hardcodes ``temperature: 0.7`` and a stingy
``max_completion_tokens: 1024``. Azure reasoning deployments (gpt-5,
gpt-5.5, o1, o3) reject any temperature except the default (1) and
need a much larger token budget because reasoning tokens count against
the same limit.

This proxy is a near-transparent forwarder:

  * Strips ``temperature`` from every chat/completions body.
  * Bumps ``max_completion_tokens`` to ``MAX_COMPLETION_TOKENS``
    (env, default 16384) when the inbound value is below it.
  * Forwards Authorization, Content-Type, and the JSON body unchanged.

Run:

    PROXY_TARGET=https://tejas-mohrgcfh-eastus2.cognitiveservices.azure.com/openai/v1 \
    PROXY_PORT=8088 \
    python scripts/azure_openai_proxy.py

Point ensemble's openai backend at the proxy:

    OPENAI_BASE_URL=http://localhost:8088
    OPENAI_API_KEY=$TEJAS_AZURE_KEY   # forwarded as Bearer; Azure v1 accepts it
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import urllib.error
import urllib.request


def _read_json(body: bytes) -> dict | None:
    if not body:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _rewrite_messages(messages: list) -> list:
    """Adapt ensemble's tool-use shape to OpenAI Chat Completions.

    Ensemble's openai backend serialises ``ChatMessage`` with only
    ``role`` and ``content`` — it drops ``tool_calls`` from assistant
    messages and emits raw ``role="tool"`` results without
    ``tool_call_id``. OpenAI rejects that conversation. Two rewrites
    keep the dialogue valid:

      * ``role="tool"`` → ``role="user"`` with a ``"Tool result:\\n"``
        prefix so the model still sees the result in-band.
      * ``role="assistant"`` with empty content → drop entirely (the
        original was a tool-only assistant turn; the tool result that
        follows is now a user message, so the assistant gap is fine).
    """
    out = []
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        role = m.get("role")
        content = m.get("content")
        if role == "tool":
            text = content if isinstance(content, str) else json.dumps(content)
            out.append({"role": "user", "content": f"Tool result:\n{text}"})
        elif role == "assistant" and (content is None or content == ""):
            continue
        else:
            out.append(m)
    return out


def _rewrite_body(body: bytes, min_max_tokens: int) -> bytes:
    parsed = _read_json(body)
    if not isinstance(parsed, dict):
        return body
    changed = False
    if "temperature" in parsed:
        parsed.pop("temperature", None)
        changed = True
    cur = parsed.get("max_completion_tokens") or parsed.get("max_tokens")
    if cur is None or (isinstance(cur, int) and cur < min_max_tokens):
        parsed["max_completion_tokens"] = min_max_tokens
        parsed.pop("max_tokens", None)
        changed = True
    messages = parsed.get("messages")
    if isinstance(messages, list):
        new_messages = _rewrite_messages(messages)
        if new_messages != messages:
            parsed["messages"] = new_messages
            changed = True
    if not changed:
        return body
    return json.dumps(parsed).encode("utf-8")


def make_handler(target: str, min_max_tokens: int, verbose: bool):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            if verbose:
                sys.stderr.write("[proxy] " + fmt % args + "\n")

        def _forward(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            if self.path.endswith("/chat/completions"):
                body = _rewrite_body(body, min_max_tokens)
            upstream_url = target.rstrip("/") + self.path
            req = urllib.request.Request(upstream_url, data=body if body else None, method=method)
            for header in ("authorization", "content-type", "accept"):
                value = self.headers.get(header)
                if value is not None:
                    req.add_header(header.title(), value)
            if body:
                req.add_header("Content-Length", str(len(body)))
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    status = resp.status
                    response_body = resp.read()
                    resp_headers = list(resp.getheaders())
            except urllib.error.HTTPError as exc:
                status = exc.code
                response_body = exc.read()
                resp_headers = list(exc.headers.items())
            except urllib.error.URLError as exc:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                err = json.dumps({"error": {"message": f"proxy upstream error: {exc.reason}"}}).encode()
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
                return
            self.send_response(status)
            forwarded = {"content-encoding", "transfer-encoding", "connection"}
            for key, value in resp_headers:
                if key.lower() in forwarded:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def do_POST(self) -> None:
            self._forward("POST")

        def do_GET(self) -> None:
            self._forward("GET")

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Azure OpenAI compatibility shim for ensemble.")
    parser.add_argument("--target", default=os.environ.get("PROXY_TARGET"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PROXY_PORT") or 8088))
    parser.add_argument("--min-max-tokens", type=int, default=int(os.environ.get("MAX_COMPLETION_TOKENS") or 16384))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if not args.target:
        parser.error("Set --target or PROXY_TARGET to the upstream Azure OpenAI v1 base URL.")
    handler = make_handler(args.target, args.min_max_tokens, args.verbose)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"[proxy] listening on http://127.0.0.1:{args.port} -> {args.target}", flush=True)
    print(f"[proxy] stripping temperature; min max_completion_tokens={args.min_max_tokens}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
