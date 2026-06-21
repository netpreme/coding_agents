from __future__ import annotations

from datasets import load_dataset

DATASET_ID = "princeton-nlp/SWE-bench_Verified"


def load() -> list[dict]:
    return [dict(row) for row in load_dataset(DATASET_ID, split="test")]
