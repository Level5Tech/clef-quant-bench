"""Store Clef's two lookup tables (token embeddings, lm_head) as int8 with one scale per row.

Neither table is used as a matrix in Clef: embeddings are looked up by token id, and the joint head only
reads the lm_head rows of each request's option tokens. In BF16 they are 4 GB of a ~10 GB checkpoint.

  python shrink_tables.py quant/clef-flash-split17 quant/clef-flash-split17-int8tables

Writes <out>/model.safetensors without the two tables and <out>/clef_head/tables_int8.safetensors with
embed_q / embed_scale / lexical_q / lexical_scale (row = q * scale). Needs the clef_vllm plugin to load.
"""
import json
import shutil
import struct
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

src, out = Path(sys.argv[1]), Path(sys.argv[2])
TABLES = {"embed": "model.language_model.embed_tokens.weight", "lexical": "lm_head.weight"}

out.mkdir(parents=True, exist_ok=True)
(out / "clef_head").mkdir(exist_ok=True)
weight_files = sorted(p for p in src.glob("*.safetensors") if "joint_head" not in p.name)
for path in src.iterdir():
    if path.is_file() and path not in weight_files:
        shutil.copy(path, out / path.name)
for path in (src / "clef_head").iterdir():
    shutil.copy(path, out / "clef_head" / path.name)

tables, before, after = {}, 0.0, 0.0
for weight_file in weight_files:  # one file, or several shards
    with open(weight_file, "rb") as f:
        metadata = json.loads(f.read(struct.unpack("<Q", f.read(8))[0])).get("__metadata__")
    tensors = load_file(weight_file)
    for name, key in TABLES.items():
        if key not in tensors:
            continue
        weight = tensors.pop(key).float()
        scale = (weight.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-8)
        q = (weight / scale).round().clamp(-127, 127).to(torch.int8)
        error = (q.float() * scale - weight).pow(2).sum().sqrt() / weight.pow(2).sum().sqrt()
        cosine = torch.nn.functional.cosine_similarity(q.float() * scale, weight, dim=1)
        tables[f"{name}_q"], tables[f"{name}_scale"] = q, scale.to(torch.bfloat16)
        print(f"{name}: {tuple(weight.shape)} relative error {float(error):.4f}, row cosine min {float(cosine.min()):.5f} "
              f"mean {float(cosine.mean()):.6f}")
    save_file(tensors, out / weight_file.name, metadata=metadata)
    before += weight_file.stat().st_size / 1e9
    after += (out / weight_file.name).stat().st_size / 1e9
    del tensors
assert len(tables) == 4, f"tables not found: {sorted(tables)}"
save_file(tables, out / "clef_head" / "tables_int8.safetensors")
index = out / "model.safetensors.index.json"
if index.exists():  # drop the two tables from the shard index
    data = json.loads(index.read_text())
    for key in TABLES.values():
        data["weight_map"].pop(key, None)
    index.write_text(json.dumps(data, indent=2))
print(f"weights {before:.2f} -> {after:.2f} GB, tables "
      f"{(out / 'clef_head' / 'tables_int8.safetensors').stat().st_size / 1e9:.2f} GB")
