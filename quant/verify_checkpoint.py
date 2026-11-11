"""Verify a quantized Clef checkpoint (structure only, no GPU, no model load).

    python -m quant.verify_checkpoint /models/clef-flash-nvfp4 --release <release-dir>

Checks: head files present and byte-identical to the release; ignored modules still hold BF16 `.weight` with no
quantization side tensors; targeted Linear modules hold quantized weights; for NVFP4, in_proj_qkv and in_proj_z
of every DeltaNet layer share one weight_global_scale (needed for vLLM's fused in_proj_qkvz GEMM); prints sizes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from quant import recipe


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--release", required=True)
    args = ap.parse_args(argv)
    from safetensors import safe_open

    ckpt, release = Path(args.checkpoint), Path(args.release)
    problems: list[str] = []

    for name in recipe.HEAD_FILES:
        a, b = ckpt / "clef_head" / name, release / name
        if not a.exists():
            problems.append(f"missing clef_head/{name}")
        elif sha256(a) != sha256(b):
            problems.append(f"clef_head/{name} differs from release")
    if not (ckpt / "joint_schema_model.py").exists():
        problems.append("missing joint_schema_model.py")

    cfg = json.loads((ckpt / "config.json").read_text())
    qcfg = cfg.get("quantization_config", {})
    print("quantization_config.format:", qcfg.get("format"), "| quant_method:", qcfg.get("quant_method"))
    print("config ignore list:", qcfg.get("ignore"))

    tensors: dict[str, tuple[str, str]] = {}  # name -> (dtype, file)
    shards = sorted(ckpt.glob("*.safetensors"))
    for shard in shards:
        with safe_open(shard, "pt") as fh:
            for key in fh.keys():
                tensors[key] = (str(fh.get_slice(key).get_dtype()), shard.name)

    modules: dict[str, set[str]] = {}
    for key in tensors:
        mod, _, leaf = key.rpartition(".")
        modules.setdefault(mod, set()).add(leaf)
    quant_leaves = {"weight_scale", "weight_packed", "weight_global_scale", "input_global_scale", "weight_zero_point"}
    n_quant = n_ignored_ok = 0
    dtype_counter: Counter = Counter()
    for mod, leaves in modules.items():
        is_linear_like = "weight" in leaves or "weight_packed" in leaves
        if not is_linear_like:
            continue
        if recipe.is_ignored(mod):
            if leaves & quant_leaves:
                problems.append(f"ignored module {mod} carries quantization tensors {sorted(leaves & quant_leaves)}")
            elif tensors[f"{mod}.weight"][0] != "BF16":
                problems.append(f"ignored module {mod} weight dtype {tensors[f'{mod}.weight'][0]} != BF16")
            else:
                n_ignored_ok += 1
        elif re.search(r"(self_attn|linear_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj|gate_proj|up_proj|down_proj)$", mod):
            if leaves & quant_leaves:
                n_quant += 1
                key = "weight_packed" if "weight_packed" in leaves else "weight"
                dtype_counter[tensors[f"{mod}.{key}"][0]] += 1
            else:
                problems.append(f"targeted module {mod} is not quantized (leaves {sorted(leaves)})")
    print(f"quantized Linear modules: {n_quant} (weight dtypes {dict(dtype_counter)}); ignored BF16 modules ok: {n_ignored_ok}")

    if qcfg.get("format") and "nvfp4" in json.dumps(qcfg).lower():
        by_layer: dict[str, dict[str, float]] = {}
        for mod in modules:
            m = re.match(r"(.*linear_attn)\.(in_proj_qkv|in_proj_z)$", mod)
            if m and "weight_global_scale" in modules[mod]:
                by_layer.setdefault(m.group(1), {})[m.group(2)] = mod
        bad = 0
        for layer, parts in by_layer.items():
            if len(parts) != 2:
                continue
            vals = []
            for mod in parts.values():
                shard = tensors[f"{mod}.weight_global_scale"][1]
                with safe_open(ckpt / shard, "pt") as fh:
                    vals.append(float(fh.get_tensor(f"{mod}.weight_global_scale").float().flatten()[0]))
            if vals[0] != vals[1]:
                bad += 1
        print(f"DeltaNet layers with in_proj_qkv/in_proj_z global scale checked: {len(by_layer)}, mismatched: {bad}")
        if bad:
            problems.append(f"{bad} DeltaNet layers have different weight_global_scale for in_proj_qkv vs in_proj_z")

    total = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
    print(f"size: {total/2**30:.2f} GiB total; safetensors shards: {sum(s.stat().st_size for s in shards)/2**30:.2f} GiB in {len(shards)} files")
    if problems:
        print("PROBLEMS:", *problems, sep="\n  ", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
