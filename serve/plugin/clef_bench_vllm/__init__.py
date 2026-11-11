"""vLLM plugin for the Clef decision models (BF16 release and llm-compressor quantizations)."""
from clef_bench_vllm.schema import ARCHITECTURE, TASK, schema_layout

__all__ = ["ARCHITECTURE", "TASK", "schema_layout", "register"]


def register() -> None:
    from vllm import ModelRegistry

    if ARCHITECTURE not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCHITECTURE, "clef_bench_vllm.model:ClefBenchForDecision")
