"""Quantize the Clef backbone with llm-compressor (0.14, transformers 5.17).

  python build_data.py                             # writes calib.jsonl / eval.jsonl
  python quantize.py --scheme FP8_DYNAMIC          # no calibration needed
  python quantize.py --scheme NVFP4                # calibrates on calib.jsonl
The joint head is not quantized; copy clef_head/ from the release next to the output.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers.helpers import FUSED_LAYER_NAMES
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--release", default="Cloudflare/clef", help="hub id or local path of the BF16 release")
ap.add_argument("--scheme", required=True, choices=["FP8_DYNAMIC", "FP8", "NVFP4"])
ap.add_argument("--calib", default="calib.jsonl")
ap.add_argument("--num-samples", type=int, default=672)
ap.add_argument("--pipeline", default="sequential")
args = ap.parse_args()

RELEASE = Path(args.release) if Path(args.release).is_dir() else Path(snapshot_download(args.release))
sys.path.insert(0, str(RELEASE))
from joint_schema_model import encode_record  # noqa: E402

IGNORE = [
    "lm_head",                      # head reads it as lexical option embeddings
    "re:.*visual.*",                # vision tower
    "re:.*linear_attn.in_proj_a$",  # GDN gate projections, 32 outputs each
    "re:.*linear_attn.in_proj_b$",
]

# vLLM fuses Qwen3.5 GDN in_proj_qkv + in_proj_z into one in_proj_qkvz GEMM, so NVFP4 needs a
# shared weight global scale for them (llm-compressor only knows q/k/v and gate/up). Without
# this vLLM takes the max scale and the qkv weights come out scaled by g_qkv / g_z.
FUSED_LAYER_NAMES.append(("in_proj_qkv", "in_proj_z"))

processor = AutoProcessor.from_pretrained(RELEASE)
model = Qwen3_5ForConditionalGeneration.from_pretrained(RELEASE, dtype=torch.bfloat16)

dataset = None
if args.scheme != "FP8_DYNAMIC":
    rows = [json.loads(l) for l in open(args.calib)][: args.num_samples]
    ids = [list(encode_record(processor.tokenizer, r["record"], processor=processor).input_ids) for r in rows]
    dataset = Dataset.from_dict({"input_ids": ids, "attention_mask": [[1] * len(x) for x in ids]})

oneshot(
    model=model,
    dataset=dataset,
    recipe=QuantizationModifier(targets="Linear", scheme=args.scheme, ignore=IGNORE),
    max_seq_length=16384,
    num_calibration_samples=len(dataset) if dataset is not None else 0,
    pipeline=args.pipeline if dataset is not None else "datafree",
    sequential_targets=["Qwen3_5DecoderLayer"],
)

out = Path("quant") / f"clef-{args.scheme.lower()}"
model.save_pretrained(out, save_compressed=True)
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
          "generation_config.json"]:
    shutil.copy(RELEASE / f, out / f)
print("saved", out)
