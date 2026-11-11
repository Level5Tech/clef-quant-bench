"""Our own llm-compressor recipe definition for Clef backbones (shared by FP8 and NVFP4).

Pure Python (no torch / llm-compressor import) so it can be unit-tested anywhere.

Quantization targets: every nn.Linear in the Qwen3.5 backbone's *language model* (attention q/k/v/o,
Gated DeltaNet in_proj_qkv / in_proj_z / out_proj, MLP gate/up/down), unless ignored below.

IGNORE (kept in BF16, identical for FP8 and NVFP4):
  lm_head                        read by the joint head as the lexical option embedding (never multiplied)
  re:.*visual.*                  the whole vision tower and merger (text-only benchmark, untested path)
  re:.*linear_attn.in_proj_a$    Gated DeltaNet gate projection (decay), 32 outputs: tiny, precision-critical
  re:.*linear_attn.in_proj_b$    Gated DeltaNet gate projection (beta),  32 outputs: tiny, precision-critical
Also unquantized by construction (not Linear layers, so QuantizationModifier never touches them):
  RMSNorm / LayerNorm weights, token embeddings (embed_tokens), conv1d in Gated DeltaNet, A_log / dt_bias.
The joint schema head (joint_head.safetensors) is not part of the backbone; it is copied byte-for-byte.

Why in_proj_a/b are excluded and the rest of the DeltaNet projections are not: the a/b projections produce the
per-head decay and write-gate of the recurrent state, so their error compounds across the sequence, and they
are 32-wide (negligible bytes). The community checkpoint kurcontko/clef-NVFP4 (rev c37a38d) makes the same
choice; this list was written independently from its model card and recipe.yaml and re-derived from the
module names of the loaded model (see quant/list_linear_modules.py).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

IGNORE: list[str] = [
    "lm_head",
    "re:.*visual.*",
    "re:.*linear_attn.in_proj_a$",
    "re:.*linear_attn.in_proj_b$",
]

# vLLM fuses Qwen3.5's Gated DeltaNet in_proj_qkv + in_proj_z into one in_proj_qkvz GEMM. A fused GEMM must
# share one NVFP4 weight global scale, but llm-compressor only knows (q,k,v) and (gate,up) as fused groups.
# Registering this pair makes llm-compressor give both weights one global scale. Without it vLLM takes the
# max of the two and silently mis-scales one of them (found by kurcontko, see third_party/ attribution).
EXTRA_FUSED_LAYER_NAMES: list[tuple[str, str]] = [("in_proj_qkv", "in_proj_z")]

# scheme name -> needs calibration data. FP8_DYNAMIC: per-channel FP8 weights + dynamic per-token FP8
# activations (data free). NVFP4: FP4 weights + FP4 activations, group size 16; activation global scales
# are calibrated from real inputs.
SCHEMES: dict[str, bool] = {"FP8_DYNAMIC": False, "NVFP4": True}

# Files copied byte-for-byte from the release next to the quantized weights.
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
                   "generation_config.json"]
HEAD_FILES = ["joint_head.safetensors", "joint_head_config.json"]
RELEASE_CODE_FILES = ["joint_schema_model.py", "LICENSE"]


def is_ignored(module_name: str) -> bool:
    """Mirror of llm-compressor's ignore matching ('re:' prefix = regex, else exact module name)."""
    for pattern in IGNORE:
        if pattern.startswith("re:"):
            if re.match(pattern[3:], module_name):
                return True
        elif module_name == pattern:
            return True
    return False


def select_calibration(primary: str | Path, fallback: str | Path) -> tuple[Path, bool]:
    """Return (path, is_fallback). Prefers the real calibration set; falls back to smoke records LOUDLY."""
    primary, fallback = Path(primary), Path(fallback)
    if primary.exists():
        return primary, False
    if fallback.exists():
        print(f"*** CALIBRATION FALLBACK: {primary} not found, using {fallback}. Development only: do NOT "
              f"publish a quant calibrated on smoke records. ***", file=sys.stderr, flush=True)
        return fallback, True
    raise FileNotFoundError(f"no calibration records: neither {primary} nor {fallback} exists")


def load_requests(path: str | Path, num_samples: int) -> list[dict[str, Any]]:
    """num_samples calibration requests (the `request` object of each PLAN.md record, plus its id).

    Task-balanced and deterministic: records are grouped by `task` (file order inside a task) and taken
    round-robin, so a file sorted by task does not bias calibration toward its first tasks.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            groups.setdefault(str(rec.get("task", "")), []).append({**rec["request"], "id": rec["id"]})
    out: list[dict[str, Any]] = []
    depth = 0
    while len(out) < num_samples and any(depth < len(g) for g in groups.values()):
        for g in groups.values():
            if depth < len(g) and len(out) < num_samples:
                out.append(g[depth])
        depth += 1
    return out
