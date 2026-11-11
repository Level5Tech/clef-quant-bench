"""Static definition of the sweep: models, quality configs, ports, GPU-memory ladders, private sets."""
from __future__ import annotations

from dataclasses import dataclass

MODELS = ("clef-flash", "clef")
PRIVATE_SETS = ("r1", "r2", "r4a", "r4b")


@dataclass(frozen=True)
class Config:
    name: str          # results dir name
    engine: str        # native | vllm | llamacpp
    fmt: str           # vllm: bf16|fp8|nvfp4 ; llamacpp: BF16|Q8_0|Q4_K_M ; native: bf16
    batch_size: int = 1
    port: int | None = None
    alias_latency_of: str | None = None   # latency identical to another config (same engine path at bs=1)
    repeat_of: str | None = None          # cross-restart repeat: same engine/flags as this config, a NEW server instance
    subset: bool = False                  # run on the first REPEAT_SUBSET_N public records only (no private sets, no latency)
    gdn_kernel: str = "torch"             # native only: Gated DeltaNet kernel, "torch" (transformers' fallback) | "fla"

    @property
    def on_subset(self) -> bool:
        """Repeats and explicit subset configs score REPEAT_SUBSET_N records per public task, nothing else."""
        return bool(self.repeat_of) or self.subset


# Requested order: ref, q8_0, q4_k_m, fp8, nvfp4, bf16-srv, gguf-bf16, ref-r2.
CONFIGS: tuple[Config, ...] = (
    Config("ref", "native", "bf16"),
    Config("q8_0", "llamacpp", "Q8_0", port=18954),
    Config("q4_k_m", "llamacpp", "Q4_K_M", port=18954),
    Config("fp8", "vllm", "fp8", port=18952),
    Config("nvfp4", "vllm", "nvfp4", port=18953),
    Config("bf16-srv", "vllm", "bf16", port=18951),
    Config("gguf-bf16", "llamacpp", "BF16", port=18954),
    # Realistic serving-noise floor: same code path as ref but batch 8 (padding + different kernels).
    Config("ref-r2", "native", "bf16", batch_size=8, alias_latency_of="ref"),
    # Cross-restart repeats (second server instance, same flags) for the engines whose results depend on the instance:
    # measured on 2026-10-16, two serial vLLM instances of the same config differ (max |dlogit| 0.94) while one instance is
    # bit-identical with itself. They run on the first REPEAT_SUBSET_N records of every public task, after their base config.
    Config("bf16-srv-r2", "vllm", "bf16", port=18951, repeat_of="bf16-srv"),
    Config("fp8-r2", "vllm", "fp8", port=18952, repeat_of="fp8"),
    Config("nvfp4-r2", "vllm", "nvfp4", port=18953, repeat_of="nvfp4"),
    Config("gguf-bf16-r2", "llamacpp", "BF16", port=18954, repeat_of="gguf-bf16"),
    # Kernel drift inside the reference itself: the same transformers code with flash-linear-attention's Triton
    # kernel for the 24 Gated DeltaNet layers instead of the pure-PyTorch fallback (which runs the chunk recurrence in
    # float32). Measured 2026-10-16 on 200 anli_r1 records: max |dlogit| 0.5625, max |dprob| 0.040, 1 argmax flip,
    # 1.48x faster on a shared GPU. `ref` stays on the fallback (deterministic across processes, higher precision).
    # A subset config, not a `-r<N>` repeat: bench.analyze reports it as a config compared with ref, not as noise floor.
    Config("ref-fla", "native", "bf16", subset=True, gdn_kernel="fla"),
)
# Restart probes, added 2026-11-03 after the sweep finished: a THIRD server instance of fp8 and nvfp4, same flags and the
# same first-200-record subsets as the -r2 repeats, run on the 27B only. They tell a few discrete start-up states (a kernel
# config picked by timing at first launch, vllm-project/vllm#58899, would make a new instance match one earlier instance
# bit for bit) apart from continuous instance-to-instance drift, and give each vLLM restart floor three pairs instead of
# one. Kept out of CONFIGS so the full-sweep job list (jobs.make_jobs) is unchanged; their 24 jobs were appended to
# sweep/jobs.jsonl by hand.
RESTART_PROBES: tuple[Config, ...] = (
    Config("fp8-r3", "vllm", "fp8", port=18952, repeat_of="fp8"),
    Config("nvfp4-r3", "vllm", "nvfp4", port=18953, repeat_of="nvfp4"),
)
REPEAT_SUBSET_N = 200
REPEAT_SUBSET_DIR = "data_r2"   # under ~/clef-quant/bench: <dir>/records/<set>.jsonl = first N records of data/records/<set>.jsonl
CONFIG_BY_NAME = {c.name: c for c in (*CONFIGS, *RESTART_PROBES)}

# Quality-run engine settings (owner decision 2026-10-16, after the concurrency study found no gain from parallel
# processes: 2 workers = 0.90x native, 0.94x llama.cpp, 1.05x vLLM).
# vLLM quality servers batch: --max-num-seqs QUALITY_MAX_NUM_SEQS and QUALITY_CONCURRENCY in-flight client requests.
# vLLM is not batch-invariant, so per-record logits move with the requests they share a batch with; the measured
# instance-to-instance drift of the same flags (max |dlogit| 0.94) already bounds noise of that kind and the -r2
# repeats measure it. The latency phase keeps one request in flight. llama.cpp's decision server is single-sequence.
QUALITY_MAX_NUM_SEQS = 16
QUALITY_CONCURRENCY = 16
# The native reference keeps transformers' pure-PyTorch Gated DeltaNet fallback (Config.gdn_kernel = "torch"): it is
# bit-identical across processes and runs the chunk recurrence in float32. flash-linear-attention is 1.48x faster but
# moves logits by up to 0.56 (see ref-fla above), so it is measured as a config, not adopted as the reference.
# Every native result records the kernel in its meta (gdn_kernel) and in engine_version.

# --gpu-memory-utilization candidates for the isolated LATENCY phase (one server alone), tried in order until
# /health. Clef-flash values are the verified README ones; Clef 27B values are README guesses plus fallbacks.
GPU_UTIL: dict[tuple[str, str], tuple[float, ...]] = {
    ("clef-flash", "bf16"): (0.25, 0.35),
    ("clef-flash", "fp8"): (0.25, 0.35),
    ("clef-flash", "nvfp4"): (0.25, 0.35),
    ("clef", "bf16"): (0.55, 0.65, 0.50),
    ("clef", "fp8"): (0.40, 0.50, 0.30),
    ("clef", "nvfp4"): (0.30, 0.40, 0.25),
}

# Quant builds needed before the 27B quantized configs (quant/build_quants.sh <model> <fmt> ...).
BUILD_ARGS: dict[str, list[str]] = {
    "fp8": [],
    "nvfp4": ["--calibration", "/bench/data/calibration.jsonl", "--num-samples", "512"],
}

# Lean ladders for concurrent QUALITY workers (server started with --max-num-seqs 1, so only ~16k tokens of KV/state
# are needed). Smaller claim = more workers fit. First rung is tried first; a refusal/OOM moves to the next rung.
GPU_UTIL_QUALITY: dict[tuple[str, str], tuple[float, ...]] = {
    ("clef-flash", "bf16"): (0.21, 0.25, 0.35),
    ("clef-flash", "fp8"): (0.16, 0.20, 0.25),
    ("clef-flash", "nvfp4"): (0.13, 0.18, 0.25),
    ("clef", "bf16"): (0.55, 0.65, 0.50),
    ("clef", "fp8"): (0.30, 0.40, 0.50),
    ("clef", "nvfp4"): (0.22, 0.30, 0.40),
}

# Concurrency, chosen by measurement (sweep/bench_concurrency.py). Max concurrent workers per (engine, model) and
# across the whole pool; the pool is additionally bounded by the unified-memory budget (units.next_launch).
WORKER_CAPS: dict[tuple[str, str], int] = {
    ("native", "clef-flash"): 1, ("vllm", "clef-flash"): 1, ("llamacpp", "clef-flash"): 1,
    ("native", "clef"): 1, ("vllm", "clef"): 1, ("llamacpp", "clef"): 1,
}
GLOBAL_SLOTS = 1
