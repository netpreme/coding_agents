"""The reverse-proxy ASGI app: forward claude-cli ↔ inference server, tee per-turn rows.

``ProxyApp(server_url, out_dir, instance_id, capture=True).build()`` returns a
Starlette app that, for each `POST /v1/messages`, sanitizes the request and
(with capture) tees the per-turn text trace to
`<out_dir>/<instance_id>/turn_traces.jsonl`; other paths stream through
unchanged (e.g. /v1/models).
"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from loguru import logger
from pipeline.proxy.parse import (
    common_prefix_len,
    extract_output_text,
    request_chunks,
    rerole_system_messages,
)
from pipeline.utils.jsonl import JsonlWriter
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


@dataclass(frozen=True)
class MessageRequest:
    """Sanitized request state for one /v1/messages call."""

    body: dict | None
    sanitized_bytes: bytes


class ProxyApp:
    """Reverse-proxy ASGI app for one problem (claude-cli ↔ inference server)."""

    def __init__(
        self,
        server_url: str,
        out_dir: Path,
        instance_id: str,
        *,
        capture: bool = False,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.instance_id = instance_id
        # With capture: tee the raw text traces (isl_new + osl) to
        # turn_traces.jsonl. `_last_turn_units` is the previous turn's request chunks, so each
        # turn's new suffix (isl_new as text) is a pure cross-turn string diff.
        self._trace_writer = (
            JsonlWriter(out_dir, "turn_traces.jsonl") if capture else None
        )
        self._last_turn_units: list[str] = []
        self._last_response_time: float | None = None

    def build(self) -> Starlette:
        return Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    endpoint=self._route,
                    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                )
            ],
            lifespan=self._lifespan,
        )

    @asynccontextmanager
    async def _lifespan(self, app: Starlette):
        async with httpx.AsyncClient(timeout=None) as client:
            app.state.client = client
            yield

    def _server_endpoint(self, request: Request) -> str:
        target = f"{self.server_url}{request.url.path}"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return target

    async def _route(self, request: Request) -> Response:
        raw_body = await request.body()
        client = request.app.state.client
        if request.method == "POST" and request.url.path == "/v1/messages":
            return await self._messages(
                client=client, request=request, raw_body=raw_body
            )
        return await self._stream_to_server(
            client=client, request=request, raw_body=raw_body
        )

    async def _messages(
        self, client: httpx.AsyncClient, request: Request, raw_body: bytes
    ) -> Response:
        """Forward a /v1/messages POST; buffer the response so we can parse
        it; write the combined request+response row; return the buffered
        response to claude-cli.

        Buffering breaks "live" streaming to claude-cli, but at concurrency=1
        that's invisible — claude still parses the SSE chunks the same way."""
        request_received_time = time.time()
        message_request = self._sanitize_request(raw_body)

        # Claude Code fires a background "sentence-case title" request once per
        # session. It shares the leading system-prompt tokens with the real
        # conversation, so forwarding it to vLLM warms the prefix cache and
        # pollutes turn-1 telemetry (isl_new < isl). Short-circuit it here with a
        # canned reply so it never reaches the engine.
        if message_request.body is not None and _is_title_request(message_request.body):
            return _canned_title_response(message_request.body.get("model") or "")

        tool_exec_ms = (
            round((request_received_time - self._last_response_time) * 1000, 2)
            if self._last_response_time is not None
            else None
        )

        server_response = await self._send_to_server(
            client=client,
            request=request,
            body=message_request.sanitized_bytes,
        )
        self._last_response_time = time.time()

        if (
            self._trace_writer is not None
            and message_request.body is not None
            and not _should_skip_telemetry(message_request.body)
        ):
            self._record_turn(
                request_received_time=request_received_time,
                tool_exec_ms=tool_exec_ms,
                body=message_request.body,
                osl_text=extract_output_text(server_response.content),
            )

        return _make_response(server_response)

    def _sanitize_request(self, raw_body: bytes) -> MessageRequest:
        try:
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise ValueError("/v1/messages body must be a JSON object")
            # claude-cli injects role:"system" messages that vLLM 400s on; the
            # top-level `system` (cached prefix) is left untouched.
            rerole_system_messages(body)
            return MessageRequest(
                body=body,
                sanitized_bytes=json.dumps(body).encode(),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("request parse error: {!r}", exc)
            return MessageRequest(body=None, sanitized_bytes=raw_body)

    async def _send_to_server(
        self,
        client: httpx.AsyncClient,
        request: Request,
        body: bytes,
    ) -> httpx.Response:
        return await client.request(
            request.method,
            self._server_endpoint(request),
            content=body,
            headers=_strip_hop_by_hop(request.headers.items()),
        )

    def _record_turn(
        self,
        request_received_time: float,
        tool_exec_ms: float | None,
        body: dict,
        osl_text: str,
    ) -> None:
        """Tee the raw text for this turn:
        isl_text     — the full input (system + tools + messages),
        isl_new_text — the input appended since the previous turn (cached
                       prefix stripped off; == isl_text on the first turn),
        osl_text     — the generated assistant text (incl. tool calls),
        tool_exec_ms — wall-clock time the agent spent executing tools between
                       the previous turn's response and this turn's request;
                       None on the first turn.
        """
        chunks = request_chunks(body)
        prefix_length = common_prefix_len(
            previous_units=self._last_turn_units,
            current_units=chunks,
        )
        self._last_turn_units = chunks

        assert self._trace_writer is not None
        self._trace_writer.write(
            instance_id=self.instance_id,
            row={
                "request_time": round(request_received_time, 3),
                "tool_exec_ms": tool_exec_ms,
                "isl_text": "\n".join(chunks),
                "isl_new_text": "\n".join(chunks[prefix_length:]),
                "osl_text": osl_text,
            },
        )

    async def _stream_to_server(
        self, client: httpx.AsyncClient, request: Request, raw_body: bytes
    ) -> Response:
        """Stream non-/v1/messages requests through unchanged (e.g. /v1/models)."""
        upstream_req = client.build_request(
            request.method,
            self._server_endpoint(request),
            content=raw_body,
            headers=_strip_hop_by_hop(request.headers.items()),
        )
        upstream_resp = await client.send(upstream_req, stream=True)
        return StreamingResponse(
            upstream_resp.aiter_raw(),
            status_code=upstream_resp.status_code,
            headers=_strip_hop_by_hop(upstream_resp.headers.items()),
            background=BackgroundTask(upstream_resp.aclose),
        )


def _canned_title_response(model: str) -> Response:
    """Anthropic-format `{"title":"Chat"}` reply for Claude Code's title request,
    returned by the proxy WITHOUT forwarding to vLLM — so the title call can't
    warm the prefix cache and skew turn-1 telemetry. Claude Code's title request
    is non-streaming, so a plain JSON message response is what it expects."""
    body = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": '{"title": "Chat"}'}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }
    return Response(content=json.dumps(body), media_type="application/json")


def _should_skip_telemetry(body: dict | None) -> bool:
    """Skip requests that vLLM handles outside the normal engine path."""
    if body is None:
        return False
    return _is_title_request(body)


def _is_title_request(body: dict) -> bool:
    """True for Claude Code's session-title request.

    Mirrors the signal vLLM's anthropic entrypoint uses to short-circuit it
    (vllm/entrypoints/anthropic/serving.py): a "generate a … sentence-case
    title" instruction in the top-level system prompt. `system` may be a
    string or a list of `{type, text}` blocks.
    """
    system = body.get("system")
    if isinstance(system, str):
        return "sentence-case title" in system
    if isinstance(system, list):
        return any(
            isinstance(block, dict)
            and "sentence-case title" in (block.get("text") or "")
            for block in system
        )
    return False


def _make_response(
    server_response: httpx.Response,
    content: bytes | None = None,
) -> Response:
    if content is None:
        content = server_response.content
    return Response(
        content=content,
        status_code=server_response.status_code,
        headers=_strip_hop_by_hop(server_response.headers.items()),
    )


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        name: value
        for name, value in headers
        if name.lower() not in _HOP_BY_HOP_HEADERS
    }
