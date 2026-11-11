"""Clef as a vLLM 0.29 pooling model: Qwen3.5 backbone in vLLM, Cloudflare's JointSchemaHead in the pooler.

Adapted from the Apache-2.0 plugin in kurcontko/clef-NVFP4 @ c37a38d (see third_party/kurcontko-clef-vllm/
ATTRIBUTION.md). Changes: written against vLLM 0.29; no int8 lookup tables (embeddings and lm_head stay exactly
as in the release: BF16 and, for quantized builds, unquantized by the recipe's ignore list); single GPU only;
head files are looked up in <model>/clef_head/ first (quantized builds) and then <model>/ (the Cloudflare
release); optional CLEF_HEAD_DTYPE=float32 runs the head in float32.

How it works: the backbone runs in vLLM and the pooler collects every request's final-norm hidden states (across
chunked prefill, if enabled) and runs the release's own JointSchemaHead on the whole prompt. Each request carries
its question/option token spans in PoolingParams.extra_kwargs["clef_questions"] (built client-side from
joint_schema_model.encode_record). The output is one flat float32 tensor: every option logit of every question,
in order. Raw logits, no rounding.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Iterable, Set
from pathlib import Path

import torch
from safetensors.torch import load_file
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.pooler import Pooler, PoolingParamsUpdate
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5ProcessingInfo
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLMultiModalProcessor
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata

from clef_bench_vllm.schema import MEM_STATS_PATH, TASK

HEAD_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}


def _import_release_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", model_dir / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def find_head_dir(model_dir: Path) -> Path:
    for candidate in (model_dir / "clef_head", model_dir):
        if (candidate / "joint_head.safetensors").exists() and (candidate / "joint_head_config.json").exists():
            return candidate
    raise FileNotFoundError(f"joint_head.safetensors / joint_head_config.json not found in {model_dir}[/clef_head]")


class ClefPooler(Pooler):
    def __init__(self, model: "ClefBenchForDecision") -> None:
        super().__init__()
        self._model = [model]  # not a submodule: the model owns (and loads) the head
        self.enable_chunked_prefill = get_current_vllm_config().scheduler_config.enable_chunked_prefill
        self._base_alloc: int | None = None  # allocator bytes at the first real request (KV pool already allocated)
        self._last_peak = -1

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {TASK}

    def get_pooling_updates(self, task: PoolingTask) -> PoolingParamsUpdate:
        return PoolingParamsUpdate(requires_token_ids=True)

    def forward(self, hidden_states: torch.Tensor, pooling_metadata: PoolingMetadata):
        model = self._model[0]
        cursor = pooling_metadata.get_pooling_cursor()
        chunks = list(torch.split(hidden_states, cursor.num_scheduled_tokens_cpu.tolist()))
        if self.enable_chunked_prefill:
            finished = cursor.is_finished().tolist()
            full: list[torch.Tensor | None] = []
            for state, chunk, done in zip(pooling_metadata.pooling_states, chunks, finished):
                state.hidden_states_cache.append(chunk if done and not state.hidden_states_cache else chunk.clone())
                if done:
                    cache = state.hidden_states_cache
                    full.append(cache[0] if len(cache) == 1 else torch.cat(cache, dim=0))
                    state.clean()
                else:
                    full.append(None)
        else:
            full = chunks

        token_ids = pooling_metadata.get_prompt_token_ids_cpu()
        outputs: list[torch.Tensor | None] = []
        for hidden, ids, params in zip(full, token_ids, pooling_metadata.pooling_params):
            if hidden is None:
                outputs.append(None)
                continue
            layout = (params.extra_kwargs or {}).get("clef_questions")
            if layout is None:  # vLLM's profiling dummy run carries no schema
                layout = [[1, [0, 1], [[0, 1], [0, 1]]]]
            if params.extra_kwargs and "clef_questions" in params.extra_kwargs and self._base_alloc is None:
                self._base_alloc = torch.cuda.memory_allocated()  # real traffic starts after KV cache allocation
                torch.cuda.reset_peak_memory_stats()
            outputs.append(model.score(hidden, ids, layout))
        self._write_memory_stats(model)
        return outputs

    def _write_memory_stats(self, model: "ClefBenchForDecision") -> None:
        if self._base_alloc is None:
            return
        peak = torch.cuda.max_memory_allocated()
        if peak == self._last_peak:
            return
        self._last_peak = peak
        gib = 2**30
        Path(MEM_STATS_PATH).write_text(json.dumps({
            "weights_gib": model.weights_bytes / gib, "base_gib": self._base_alloc / gib, "peak_gib": peak / gib,
            "activation_peak_gib": max(0, peak - self._base_alloc) / gib}))


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class ClefBenchForDecision(Qwen3_5ForConditionalGeneration):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("clef_bench_vllm supports tensor_parallel_size=1 only")
        self.model_dir = Path(vllm_config.model_config.model)
        self.release = _import_release_module(self.model_dir)
        self.head_dir = find_head_dir(self.model_dir)
        head_config = json.loads((self.head_dir / "joint_head_config.json").read_text())
        head_config.pop("requires_clef_vllm", None)
        self.head = self.release.JointSchemaHead(**head_config)
        self.head_dtype = HEAD_DTYPES[os.environ.get("CLEF_HEAD_DTYPE", "bfloat16")]
        self.pooler = ClefPooler(self)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = super().load_weights(weights)
        state = load_file(self.head_dir / "joint_head.safetensors")
        self.head.load_state_dict(state, strict=True)
        lm_head = self.language_model.lm_head.weight
        # Cloudflare's loader casts the head to the backbone dtype (BF16); CLEF_HEAD_DTYPE=float32 overrides.
        self.head.to(device=lm_head.device, dtype=self.head_dtype).eval()
        self.lexical_weight = lm_head.data  # the head only reads rows of option tokens
        self.weights_bytes = torch.cuda.memory_allocated()  # weights + head, before the KV/state cache exists
        return loaded | {f"head.{name}" for name in state}

    @torch.inference_mode()
    def score(self, hidden: torch.Tensor, ids: torch.Tensor, layout: list) -> torch.Tensor:
        release = self.release
        questions = tuple(
            release.EncodedQuestion(
                question_id=str(index),
                question_type=int(type_id),
                question_span=tuple(question_span),
                option_spans=tuple(tuple(span) for span in option_spans),
                option_ids=tuple(str(i) for i in range(len(option_spans))),
            )
            for index, (type_id, question_span, option_spans) in enumerate(layout)
        )
        record = release.EncodedRecord(input_ids=tuple(), questions=questions, record_id="vllm")
        device = hidden.device
        length = hidden.shape[0]
        # The head reads lm_head rows only at option-span tokens: gather just those rows and remap the span
        # token ids into that compact table (identical values to indexing the full matrix).
        ids = ids.cpu()
        positions = torch.cat([torch.arange(start, end) for _, _, spans in layout for start, end in spans])
        unique, inverse = torch.unique(ids[positions].long(), return_inverse=True)
        lexical = self.lexical_weight[unique.to(self.lexical_weight.device)].to(self.head_dtype)
        ids = torch.zeros(ids.shape, dtype=torch.long)
        ids[positions] = inverse
        logits = self.head(
            hidden.unsqueeze(0).to(self.head_dtype),
            ids.to(device).unsqueeze(0),
            torch.ones((1, length), dtype=torch.long, device=device),
            [record],
            lexical,
        )[0]
        return torch.cat([question_logits.float() for question_logits in logits])
