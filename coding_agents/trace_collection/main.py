"""Benchmark runner."""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from coding_agents.serving import close_gpu_handles, open_gpu_handles, read_gpu_stats
from coding_agents.trace_collection.pipeline.harness import DEFAULT_TIMEOUT_S, run_task
from coding_agents.trace_collection.pipeline.datasets import get_dataset
from coding_agents.trace_collection.pipeline.inference_servers import (
    BACKEND_NAMES,
    get_inference_server,
)
from coding_agents.trace_collection.pipeline.sandbox import Sandbox
from coding_agents.trace_collection.pipeline.utils.config import (
    save_config,
    save_session_config,
)

ROOT_DIR = Path(__file__).resolve().parent
SERVER_URL = "http://localhost:8000"
PROXY_PORT = 8001
ANTHROPIC_URL = "https://api.anthropic.com"
WAIT_TIME = 0.3


def main() -> int:
    logger.remove()
    logger.add(
        lambda msg: tqdm.write(str(msg), end="", file=sys.stderr),
        colorize=True,
    )
    args = _parse_args()
    save_dir = _resolve_save_dir(args.resume)
    save_dir.mkdir(parents=True, exist_ok=True)

    inference_server = get_inference_server(
        args.backend,
        server_url=SERVER_URL,
        proxy_port=PROXY_PORT,
        anthropic_url=ANTHROPIC_URL,
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tool_call_parser=args.tool_call_parser,
    )

    tasks, dataset_id, skipped_count = get_dataset(
        args.dataset,
        resume_path=save_dir / "runs",
        num_samples=args.num_samples,
    )
    logger.info(
        "{} problems pending ({} skipped) to {}", len(tasks), skipped_count, save_dir
    )

    save_config(
        save_dir=save_dir,
        runtime=inference_server.config(),
        args=vars(args),
        dataset_name=dataset_id,
        pending_count=len(tasks),
        skipped_solved_count=skipped_count,
        run_start_ts=time.time(),
    )

    turn_count = 0
    latest_row: dict = {}
    gpu_handles = open_gpu_handles()

    progress_bar = tqdm(
        tasks,
        desc="solving",
        unit="problem",
        dynamic_ncols=True,
        file=sys.stderr,
        position=0,
    )
    metrics_bar = tqdm(
        bar_format="{desc}",
        desc="  (waiting for first turn…)",
        dynamic_ncols=True,
        file=sys.stderr,
        position=1,
    )

    def _update_live_metrics(row: dict) -> None:
        nonlocal turn_count
        turn_count += 1
        latest_row.update(row)
        metrics_bar.set_description(
            _format_turn_metrics(
                turn_count=turn_count, row=latest_row, gpu_handles=gpu_handles
            )
        )

    try:
        with logging_redirect_tqdm():
            for task in progress_bar:
                turn_count = 0
                latest_row.clear()
                metrics_bar.set_description("  (waiting for first turn…)")
                logger.info(
                    "{0} ({1} @ {2})".format(
                        task["instance_id"], task["repo"], task["base_commit"][:5]
                    )
                )
                with (
                    inference_server.session(
                        save_dir=save_dir,
                        instance_id=task["instance_id"],
                        capture=args.capture,
                        fn=_update_live_metrics,
                    ) as server,
                    Sandbox(
                        save_dir=save_dir, prefix=f"{task['instance_id']}."
                    ) as sandbox,
                ):
                    time.sleep(WAIT_TIME)
                    exit_code = run_task(
                        task=task,
                        sandbox_dir=sandbox.dir,
                        model=server.model,
                        base_url=server.base_url,
                        timeout_s=args.timeout,
                        oauth=server.oauth,
                        runs_dir=server.transcript_runs_dir,
                    )
                    server_runtime = server.runtime()

                save_session_config(
                    save_dir=save_dir,
                    task=task,
                    runtime=server_runtime,
                    start_ts=sandbox.start_ts,
                    end_ts=sandbox.end_ts,
                    exit_code=exit_code,
                )
    finally:
        progress_bar.close()
        metrics_bar.close()
        if gpu_handles:
            close_gpu_handles()

    return 0


def _format_turn_metrics(turn_count: int, row: dict, gpu_handles: list) -> str:
    """Format a one-line metrics description from the latest MetricsScraper row."""
    parts = [f"turn={turn_count}"]
    isl = row.get("isl")
    isl_new = row.get("isl_new")
    if isl is not None:
        parts.append(f"isl={isl:.0f}")
    if row.get("osl") is not None:
        parts.append(f"osl={row['osl']:.0f}")
    if isl_new is not None:
        parts.append(f"isl_new={isl_new:.0f}")
    if isl and isl_new is not None and isl > 0:
        parts.append(f"prefix_hit={(isl - isl_new) / isl * 100:.1f}%")
    if row.get("prefill_ms") is not None:
        parts.append(f"prefill={row['prefill_ms']:.0f}ms")
    if row.get("decode_ms") is not None:
        parts.append(f"decode={row['decode_ms']:.0f}ms")
    gpu_util, gpu_mem = read_gpu_stats(gpu_handles)
    if gpu_util:
        parts.append(gpu_util)
    if gpu_mem:
        parts.append(gpu_mem)
    return "  " + "  ".join(parts)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default="vllm",
        help=f"inference backend; available: {', '.join(BACKEND_NAMES)} (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset",
        default="pro",
        help="pro or verfied to use SWE Bench dataset",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        default=False,
        dest="capture",
        help="capture per-turn raw text traces in turn_traces.jsonl (vLLM only)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        dest="num_samples",
        help="run at most N pending problems (e.g. --num-samples 1)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="reuse <SAVE_DIR>, skipping problems already solved in it",
    )
    parser.add_argument("--model", default=None, help="model to serve")
    parser.add_argument(
        "--tensor-parallel", "-tp", dest="tensor_parallel_size", type=int, default=1
    )
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tool-call", dest="tool_call_parser", default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"wall-clock cap (seconds) per claude session (default: {DEFAULT_TIMEOUT_S})",
    )
    return parser.parse_args()


def _resolve_save_dir(resume: Path | None) -> Path:
    if resume is not None:
        return resume.expanduser().resolve()
    return ROOT_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    sys.exit(main())
