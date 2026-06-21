"""The reverse-proxy ASGI app: forward claude-cli ↔ inference server."""

from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from coding_agents.trace_collection.pipeline.proxy.parse import (
    extract_output_text,
    rerole_system_messages,
)
from coding_agents.trace_collection.pipeline.proxy.recorder import (
    MessageTrace,
    TraceRecorder,
)
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
    body: dict | None
    sanitized_bytes: bytes


class ProxyApp:
    def __init__(
        self, server_url: str, out_dir: Path, instance_id: str, *, capture: bool = False
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.instance_id = instance_id
        self._recorder = TraceRecorder(out_dir, instance_id, capture=capture)

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
        request_received_time = __import__("time").time()
        message_request = self._sanitize_request(raw_body)
        if message_request.body is not None and _is_title_request(message_request.body):
            return _canned_title_response(message_request.body.get("model") or "")
        tool_exec_ms = self._recorder.note_response_time()
        server_response = await self._send_to_server(
            client=client, request=request, body=message_request.sanitized_bytes
        )
        if message_request.body is not None:
            self._recorder.record_message(
                MessageTrace(
                    request_time=request_received_time,
                    tool_exec_ms=tool_exec_ms,
                    body=message_request.body,
                    osl_text=extract_output_text(server_response.content),
                )
            )
        return _make_response(server_response)

    def _sanitize_request(self, raw_body: bytes) -> MessageRequest:
        try:
            body = json.loads(raw_body)
            if not isinstance(body, dict):
                raise ValueError("/v1/messages body must be a JSON object")
            rerole_system_messages(body)
            return MessageRequest(body=body, sanitized_bytes=json.dumps(body).encode())
        except (json.JSONDecodeError, ValueError):
            return MessageRequest(body=None, sanitized_bytes=raw_body)

    async def _send_to_server(
        self, client: httpx.AsyncClient, request: Request, body: bytes
    ) -> httpx.Response:
        return await client.request(
            request.method,
            self._server_endpoint(request),
            content=body,
            headers=_strip_hop_by_hop(request.headers.items()),
        )

    async def _stream_to_server(
        self, client: httpx.AsyncClient, request: Request, raw_body: bytes
    ) -> Response:
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


def _is_title_request(body: dict) -> bool:
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
    server_response: httpx.Response, content: bytes | None = None
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
