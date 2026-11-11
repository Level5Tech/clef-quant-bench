"""Print which Linear modules of a Clef release the recipe quantizes vs ignores (meta device: no weights loaded).

    python -m quant.list_linear_modules /path/to/release
"""
from __future__ import annotations

import re
import sys
from collections import Counter

from quant.recipe import is_ignored


def generic(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


def main(path: str) -> None:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3_5ForConditionalGeneration

    config = AutoConfig.from_pretrained(path)
    with init_empty_weights():
        model = Qwen3_5ForConditionalGeneration._from_config(config, dtype=torch.bfloat16)
    quantized, ignored = Counter(), Counter()
    params_q = params_i = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            n = module.weight.numel()
            if is_ignored(name):
                ignored[generic(name)] += 1
                params_i += n
            else:
                quantized[generic(name)] += 1
                params_q += n
    print("QUANTIZED Linear groups (pattern: count)")
    for k, v in sorted(quantized.items()):
        print(f"  {v:4d}  {k}")
    print("IGNORED Linear groups (kept BF16)")
    for k, v in sorted(ignored.items()):
        print(f"  {v:4d}  {k}")
    total = params_q + params_i
    print(f"Linear params quantized: {params_q/1e9:.3f}B ({100*params_q/total:.1f}% of Linear), ignored {params_i/1e9:.3f}B")


if __name__ == "__main__":
    main(sys.argv[1])
