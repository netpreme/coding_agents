# Coding Agents Experiments/Applications

Coding agents take turns to carry out tasks. Using Claude Code as the harness, the repo contains code to:
* collect per-turn token counts, inference metrics, inter-turn CPU execution time (harness wall-clock time between inference calls), and tool call distributions from live coding agent sessions
* replay captured coding agent sessions from the above using an inference engine to benchmark throughput, latency, efficiency and power consumption 

## Modules

### [`trace_collection`](coding_agents/trace_collection/README.md)

Runs coding agents against a SWE Bench dataset and captures per-turn token counts (ISL, OSL, ISL_new), inter-turn tool execution time, and server-side inference metrics via a transparent proxy. Produces a results directory of captured sessions that can be used directly for analysis or replayed.

### [`trace_replay`](coding_agents/trace_replay/README.md)

Replays captured sessions against a fresh inference engine to measure production throughput and latency under realistic concurrent load — without running live agents. The primary metrics are agents per megawatt and agents per GPU, with supporting breakdowns of TTFT, ITL, prefill/decode time, prefix cache hit rate, and output token throughput.

## Installation

Install prerequisites and set up the Python virtual environment.

```bash
./install.sh
```
