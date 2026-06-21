from coding_agents.trace_collection.pipeline.inference_servers.anthropic.backend import (
    AnthropicBackend,
)
from coding_agents.trace_collection.pipeline.inference_servers.vllm.backend import (
    VllmBackend,
)

BACKEND_NAMES = ["vllm", "anthropic"]


def get_inference_server(
    name: str,
    *,
    server_url: str,
    proxy_port: int,
    anthropic_url: str,
    model: str | None,
    tensor_parallel_size: int | None,
    max_model_len: int | None,
    gpu_memory_utilization: float | None,
    tool_call_parser: str | None,
) -> VllmBackend | AnthropicBackend:
    if name in {"vllm", "local-vllm"}:
        return VllmBackend(
            server_url=server_url,
            proxy_port=proxy_port,
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tool_call_parser=tool_call_parser,
        )
    if name in {"anthropic", "claude"}:
        return AnthropicBackend(anthropic_url=anthropic_url, model=model)
    raise ValueError(
        "unknown backend: "
        f"{name!r}; choose from: anthropic, claude, local-vllm, vllm"
    )
