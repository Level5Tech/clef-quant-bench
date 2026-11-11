"""Quantize a Clef release backbone with llm-compressor (our recipe, quant/recipe.py).

  FP8 W8A8 (FP8_DYNAMIC, data free):
    python -m quant.quantize --release <release-dir> --scheme FP8_DYNAMIC --out /models/clef-flash-fp8
  NVFP4 W4A4 (calibrated on Cloudflare-encoded records):
    python -m quant.quantize --release <release-dir> --scheme NVFP4 --out /models/clef-flash-nvfp4 \
        --calibration /bench/data/calibration.jsonl --num-samples 512

Run through quant/run_quant.sh (NGC vllm 26.09 container + llm-compressor 0.14.0 venv). The calibration
records are rendered with Cloudflare's own encode_record (joint_schema_model.py from the release), so the
activations seen during calibration are the activations Clef sees in production: system prompt, STATE,
SCHEMA FIELDS, and the assistant prefix, in that order.

The joint schema head is not quantized: joint_head.safetensors / joint_head_config.json are copied
byte-for-byte into <out>/clef_head/ (a subfolder, so vLLM's weight loader never mistakes them for backbone
weights), together with joint_schema_model.py and the tokenizer/processor files.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

from quant import recipe

FORMAT_SUFFIX = {"FP8_DYNAMIC": "fp8", "NVFP4": "nvfp4"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def import_release_module(release: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", release / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True, help="local Cloudflare release dir (snapshot) of clef / clef-flash")
    ap.add_argument("--scheme", required=True, choices=sorted(recipe.SCHEMES))
    ap.add_argument("--out", required=True, help="output dir, e.g. /models/clef-flash-fp8")
    ap.add_argument("--calibration", default="/bench/data/calibration.jsonl")
    ap.add_argument("--calibration-fallback", default="/bench/data/smoke/smoke.jsonl",
                    help="development fallback used (loudly) when --calibration does not exist")
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-length", type=int, default=16384)
    ap.add_argument("--pipeline", default="sequential", help="llm-compressor pipeline for calibrated schemes")
    args = ap.parse_args(argv)

    import llmcompressor
    import compressed_tensors
    import torch
    import transformers
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.observers.helpers import FUSED_LAYER_NAMES
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    release = Path(args.release)
    out = Path(args.out)
    needs_calibration = recipe.SCHEMES[args.scheme]
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    for pair in recipe.EXTRA_FUSED_LAYER_NAMES:
        if pair not in FUSED_LAYER_NAMES:
            FUSED_LAYER_NAMES.append(pair)

    processor = AutoProcessor.from_pretrained(release)
    release_module = import_release_module(release)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(release, dtype=torch.bfloat16)

    calibration_info: dict = {"used": False}
    dataset = None
    if needs_calibration:
        cal_path, is_fallback = recipe.select_calibration(args.calibration, args.calibration_fallback)
        requests = recipe.load_requests(cal_path, args.num_samples)
        ids = [list(release_module.encode_record(processor.tokenizer, r, max_length=args.max_seq_length,
                                                 processor=processor).input_ids) for r in requests]
        dataset = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})
        calibration_info = {
            "used": True, "path": str(cal_path), "is_dev_fallback": is_fallback, "sha256": sha256(cal_path),
            "samples": len(ids), "tokens_total": sum(map(len, ids)),
            "tokens_min": min(map(len, ids)), "tokens_max": max(map(len, ids)),
            "encoder": "Cloudflare joint_schema_model.encode_record (release code)",
        }
        print(f"calibration: {calibration_info}", flush=True)

    modifier = QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=list(recipe.IGNORE))
    oneshot(
        model=model,
        dataset=dataset,
        recipe=modifier,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(dataset) if dataset is not None else 0,
        pipeline=args.pipeline if needs_calibration else "datafree",
        sequential_targets=["Qwen3_5DecoderLayer"],
        shuffle_calibration_samples=False,
    )
    model.save_pretrained(out, save_compressed=True)

    # Release files copied byte-for-byte. The head goes in clef_head/ so vLLM does not load it as backbone weights.
    for name in recipe.TOKENIZER_FILES + recipe.RELEASE_CODE_FILES:
        if (release / name).exists():
            shutil.copy2(release / name, out / name)
    (out / "clef_head").mkdir(exist_ok=True)
    for name in recipe.HEAD_FILES:
        shutil.copy2(release / name, out / "clef_head" / name)

    meta = {
        "scheme": args.scheme,
        "format": FORMAT_SUFFIX[args.scheme],
        "release": str(release),
        "ignore": recipe.IGNORE,
        "extra_fused_layer_names": recipe.EXTRA_FUSED_LAYER_NAMES,
        "calibration": calibration_info,
        "head_sha256": {n: sha256(release / n) for n in recipe.HEAD_FILES},
        "versions": {"llmcompressor": llmcompressor.__version__, "compressed_tensors": compressed_tensors.__version__,
                     "transformers": transformers.__version__, "torch": torch.__version__},
        "pipeline": args.pipeline if needs_calibration else "datafree",
        "seconds": round(time.time() - started, 1),
    }
    (out / "clef_quant_recipe.json").write_text(json.dumps(meta, indent=2))
    print("saved", out, json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
