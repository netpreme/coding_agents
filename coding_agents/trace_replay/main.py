"""Entry point — wire sessions, vLLM server, metrics, and the replay engine."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

from .engine import replay_traces
from .trace_loader import build_server_config_from_capture, load_and_shuffle_sessions
from .summary import compute_and_write_summary, save_config, write_summary_csv

_DEFAULT_SEED = 0


def main() -> None:
    args = _parse_cli_args()

    trace_path = args.trace_path.expanduser().resolve()
    if not trace_path.exists():
        raise FileNotFoundError(f"--trace_path not found: {trace_path}")

    capture_config_path = trace_path / "run_config.json"
    if not capture_config_path.exists():
        raise FileNotFoundError(
            f"capture run_config.json not found: {capture_config_path}"
        )

    capture_config = json.loads(capture_config_path.read_text())
    sessions = load_and_shuffle_sessions(results_dir=trace_path, seed=_DEFAULT_SEED)
    logger.info(
        "loaded {} sessions, {} turns total",
        len(sessions),
        sum(len(s["turns"]) for s in sessions),
    )

    config = build_server_config_from_capture(capture_config)
    gpus = config.tensor_parallel_size
    gpu_name = (capture_config.get("gpu") or {}).get("name")
    duration_s = args.duration * 60

    output_dir = args.output_dir or (Path(__file__).resolve().parent / "results")
    save_dir = output_dir / _output_dir_name()
    save_dir.mkdir(parents=True, exist_ok=True)

    phase_id = f"concurrency_{args.concurrency}"
    phase_dir = save_dir / "runs" / phase_id
    phase_dir.mkdir(parents=True, exist_ok=True)

    save_config(
        save_dir=save_dir,
        trace_path=trace_path,
        capture_config=capture_config,
        cli_args=vars(args),
        duration_s=duration_s,
        gpus=gpus,
        gpu_name=gpu_name,
    )

    logger.info(
        "replay phase: concurrency={}, duration={:.0f}s", args.concurrency, duration_s
    )
    stats, power_watts = replay_traces(
        sessions=sessions,
        config=config,
        save_dir=save_dir,
        phase_dir=phase_dir,
        phase_id=phase_id,
        concurrency=args.concurrency,
        duration_s=duration_s,
    )

    summary = compute_and_write_summary(
        phase_dir=phase_dir,
        stats=stats,
        concurrency=args.concurrency,
        duration_s=duration_s,
        gpus=gpus,
        gpu_name=gpu_name,
        power_watts=power_watts,
    )
    write_summary_csv(save_dir=save_dir, summaries=[summary])
    logger.info("results written to {}", save_dir)


def _output_dir_name() -> str:
    return "replay_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay captured traces against an inference engine with concurrent virtual agents."
    )
    parser.add_argument(
        "--trace_path",
        type=Path,
        required=True,
        help="results/<stamp>/ directory from a prior capture run",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="number of concurrent virtual agents (default: 8)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="replay duration in minutes (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="directory for replay results (default: replay/results/)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
