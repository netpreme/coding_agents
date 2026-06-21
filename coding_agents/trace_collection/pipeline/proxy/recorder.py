from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from coding_agents.trace_collection.pipeline.proxy.parse import (
    common_prefix_len,
    request_chunks,
)
from coding_agents.trace_collection.pipeline.utils.config import JsonlWriter


@dataclass(frozen=True)
class MessageTrace:
    request_time: float
    tool_exec_ms: float | None
    body: dict
    osl_text: str


class TraceRecorder:
    def __init__(
        self, out_dir: Path, instance_id: str, *, capture: bool = False
    ) -> None:
        self.instance_id = instance_id
        self._writer = JsonlWriter(out_dir, "turn_traces.jsonl") if capture else None
        self._last_turn_units: list[str] = []
        self._last_response_time: float | None = None

    def note_response_time(self) -> float | None:
        now = time.time()
        tool_exec_ms = (
            round((now - self._last_response_time) * 1000, 2)
            if self._last_response_time is not None
            else None
        )
        self._last_response_time = now
        return tool_exec_ms

    def record_message(self, trace: MessageTrace) -> None:
        if self._writer is None:
            return
        chunks = request_chunks(trace.body)
        prefix_length = common_prefix_len(
            previous_units=self._last_turn_units,
            current_units=chunks,
        )
        self._last_turn_units = chunks
        self._writer.write(
            instance_id=self.instance_id,
            row={
                "request_time": round(trace.request_time, 3),
                "tool_exec_ms": trace.tool_exec_ms,
                "isl_text": "\n".join(chunks),
                "isl_new_text": "\n".join(chunks[prefix_length:]),
                "osl_text": trace.osl_text,
            },
        )
