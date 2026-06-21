from __future__ import annotations

import json

from datasets import load_dataset

DATASET_ID = "ScaleAI/SWE-bench_Pro"


def load() -> list[dict]:
    rows = [dict(row) for row in load_dataset(DATASET_ID, split="test")]
    return [_normalize(row) for row in rows]


def _normalize(row: dict) -> dict:
    row = dict(row)
    parts = [
        _filter_quotation_marks(row["problem_statement"]),
        "# Requirements\n" + _filter_quotation_marks(row["requirements"]),
    ]
    interface = _filter_quotation_marks(row["interface"])
    if interface and interface != "-":
        parts.append("# Interface\n" + interface)
    row["problem_statement"] = "\n\n".join(parts)
    return row


def _filter_quotation_marks(value: str) -> str:
    if value.startswith('"') and value.endswith('"'):
        return json.loads(value)
    return value
