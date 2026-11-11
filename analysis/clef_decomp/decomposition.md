# Noise-floor decomposition: clef

Comparison unit: first 200 records of each of 12 public tasks (2400 records, 2400 questions). Bootstrap: 2000 paired resamples over records, seed 20261015, percentile 95% CI; one index matrix is shared by every row, so differences are paired. Flip = top-1 option differs between the two configs (rates are shares of questions; one question per record on these tasks, so questions = records). TVD = mean total-variation distance of the option distributions. Bit-identical = all logits equal (logits present on both sides) or all stored probabilities equal (llama.cpp, which returns full-precision probabilities and no logits). max |dlogit| raw includes any constant shift of the logit scale, which softmax ignores (it dominates NVFP4 pairs); the centred value removes it. Exact = Clopper-Pearson interval (questions treated as independent), shown because the percentile bootstrap collapses to a point at 0 flips.

## Restart of the same vLLM flags (new server instance)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16-srv vs bf16-srv-r2 | vLLM restart, BF16 | 2400 | 4 | 0.17% [0.04, 0.33] | [0.05, 0.43] | 0.00257 [0.00234, 0.00282] | 0.1343 | 0.531 | 0.498 | 161 (6.7%) [logits] |
| fp8 vs fp8-r2 | vLLM restart, FP8 | 2400 | 0 | 0.00% [0.00, 0.00] | [0.00, 0.15] | 0.00055 [0.00036, 0.00078] | 0.1170 | 0.469 | 0.388 | 2288 (95.3%) [logits] |
| nvfp4 vs nvfp4-r2 | vLLM restart, NVFP4 | 2400 | 22 | 0.92% [0.54, 1.33] | [0.58, 1.38] | 0.00710 [0.00581, 0.00856] | 0.5891 | 1.855 | 2.060 | 2087 (87.0%) [logits] |

## Restart of llama.cpp (new server instance)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| gguf-bf16 vs gguf-bf16-r2 | llama.cpp restart, BF16 (N5) | 2400 | 0 | 0.00% [0.00, 0.00] | [0.00, 0.15] | 0.00000 [0.00000, 0.00000] | 0.0000 | - | - | 2400 (100.0%) [probs] |

## Native rerun at batch 8 (rerun + batch-size change, NOT a pure restart)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| ref vs ref-r2 | native rerun at batch 8 vs batch 1 (a rerun AND a batch-size change, not a pure restart) | 2400 | 15 | 0.62% [0.33, 0.96] | [0.35, 1.03] | 0.00491 [0.00451, 0.00536] | 0.2140 | 1.004 | 0.818 | 62 (2.6%) [logits] |

## Kernel selection (Gated DeltaNet: PyTorch fallback vs flash-linear-attention)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| ref vs ref-fla | Gated DeltaNet kernel: PyTorch fallback vs flash-linear-attention (N4) | 2400 | 14 | 0.58% [0.29, 0.88] | [0.32, 0.98] | 0.00521 [0.00481, 0.00565] | 0.1868 | 1.684 | 0.847 | 48 (2.0%) [logits] |

## Engine change at BF16

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| ref vs bf16-srv | engine at BF16: transformers -> vLLM (N3) | 2400 | 13 | 0.54% [0.25, 0.83] | [0.29, 0.92] | 0.00610 [0.00565, 0.00659] | 0.1901 | 1.031 | 0.608 | 51 (2.1%) [logits] |
| ref vs gguf-bf16 | engine at BF16: transformers -> llama.cpp (N3) | 2400 | 16 | 0.67% [0.33, 1.00] | [0.38, 1.08] | 0.00513 [0.00473, 0.00556] | 0.1754 | - | - | 0 (0.0%) [probs] |

## Quantization inside one engine

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16-srv vs fp8 | FP8 vs BF16 inside vLLM (N1) | 2400 | 39 | 1.62% [1.12, 2.12] | [1.16, 2.21] | 0.01881 [0.01749, 0.02023] | 0.5597 | 1.879 | 1.373 | 1 (0.0%) [logits] |
| bf16-srv vs nvfp4 | NVFP4 vs BF16 inside vLLM (N2) | 2400 | 136 | 5.67% [4.79, 6.58] | [4.78, 6.67] | 0.05450 [0.05104, 0.05793] | 0.6283 | 3.703 | 2.630 | 0 (0.0%) [logits] |
| gguf-bf16 vs q8_0 | Q8_0 vs BF16 inside llama.cpp (N3 8-bit) | 2400 | 25 | 1.04% [0.67, 1.46] | [0.68, 1.53] | 0.00846 [0.00777, 0.00925] | 0.3521 | - | - | 0 (0.0%) [probs] |
| gguf-bf16 vs q4_k_m | Q4_K_M vs BF16 inside llama.cpp (N2) | 2400 | 84 | 3.50% [2.79, 4.29] | [2.80, 4.32] | 0.03697 [0.03436, 0.03945] | 0.6292 | - | - | 0 (0.0%) [probs] |

## Every config against ref (for completeness)

| pair | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|
| ref vs bf16-srv | 2400 | 13 | 0.54% [0.25, 0.83] | [0.29, 0.92] | 0.00610 [0.00565, 0.00659] | 0.1901 | 1.031 | 0.608 | 51 (2.1%) [logits] |
| ref vs fp8 | 2400 | 33 | 1.38% [0.92, 1.83] | [0.95, 1.93] | 0.01906 [0.01768, 0.02061] | 0.4684 | 1.973 | 1.336 | 1 (0.0%) [logits] |
| ref vs nvfp4 | 2400 | 137 | 5.71% [4.83, 6.62] | [4.81, 6.71] | 0.05488 [0.05146, 0.05841] | 0.6176 | 3.578 | 2.579 | 0 (0.0%) [logits] |
| ref vs gguf-bf16 | 2400 | 16 | 0.67% [0.33, 1.00] | [0.38, 1.08] | 0.00513 [0.00473, 0.00556] | 0.1754 | - | - | 0 (0.0%) [probs] |
| ref vs q8_0 | 2400 | 23 | 0.96% [0.58, 1.33] | [0.61, 1.43] | 0.00939 [0.00859, 0.01028] | 0.3307 | - | - | 0 (0.0%) [probs] |
| ref vs q4_k_m | 2400 | 84 | 3.50% [2.79, 4.25] | [2.80, 4.32] | 0.03703 [0.03442, 0.03960] | 0.6241 | - | - | 0 (0.0%) [probs] |
| ref vs ref-r2 | 2400 | 15 | 0.62% [0.33, 0.96] | [0.35, 1.03] | 0.00491 [0.00451, 0.00536] | 0.2140 | 1.004 | 0.818 | 62 (2.6%) [logits] |
| ref vs ref-fla | 2400 | 14 | 0.58% [0.29, 0.88] | [0.32, 0.98] | 0.00521 [0.00481, 0.00565] | 0.1868 | 1.684 | 0.847 | 48 (2.0%) [logits] |
| ref vs bf16-srv-r2 | 2400 | 15 | 0.62% [0.33, 0.96] | [0.35, 1.03] | 0.00639 [0.00592, 0.00692] | 0.1809 | 1.277 | 0.850 | 48 (2.0%) [logits] |
| ref vs fp8-r2 | 2400 | 33 | 1.38% [0.92, 1.83] | [0.95, 1.93] | 0.01911 [0.01775, 0.02063] | 0.4684 | 1.973 | 1.336 | 1 (0.0%) [logits] |
| ref vs nvfp4-r2 | 2400 | 139 | 5.79% [4.92, 6.71] | [4.89, 6.80] | 0.05475 [0.05127, 0.05837] | 0.6176 | 3.578 | 2.579 | 0 (0.0%) [logits] |
| ref vs gguf-bf16-r2 | 2400 | 16 | 0.67% [0.33, 1.00] | [0.38, 1.08] | 0.00513 [0.00473, 0.00556] | 0.1754 | - | - | 0 (0.0%) [probs] |

## Pre-registered tests (PLAN.md, N1-N5)

Each row is a paired-bootstrap interval of a DIFFERENCE of flip rates on the same records (percentage points). `strat` = same test with records resampled inside each task. Outcomes are reported as found.

### N1 FP8 inside restart noise

Criterion: supported if FP8's flip rate against bf16-srv is NOT significantly larger than the bf16-srv restart flip rate, i.e. the 95% CI of the difference does not lie above 0.

| difference | flips | diff pp [95% CI] | strat diff pp [95% CI] | excludes 0 |
|---|---|---|---|---|
| flip(bf16-srv|fp8) - flip(bf16-srv|bf16-srv-r2) | 39 vs 4 | +1.458 [0.958, 1.958] | [0.958, 2.000] | yes |

N1 supported: False.

Supplementary, not pre-registered (FP8 vs BF16 minus FP8 against its own restart (fp8 vs fp8-r2)): 39 vs 0 flips, diff +1.62 pp [1.12, 2.12], excludes 0: True.

### N2 NVFP4 and Q4_K_M exceed every non-quantization source

Criterion: supported if every quantization flip rate (NVFP4 vs bf16-srv, Q4_K_M vs gguf-bf16) exceeds every non-quantization source with a 95% CI of the difference that excludes 0 (lower bound > 0).

| quantization | minus source | flips | diff pp [95% CI] | strat [95% CI] | Bonferroni (k=16) [CI] | lower > 0 |
|---|---|---|---|---|---|---|
| bf16-srv|nvfp4 | bf16-srv|bf16-srv-r2 | 136 vs 4 | +5.50 [4.62, 6.42] | [4.58, 6.33] | [4.06, 6.99] | yes |
| bf16-srv|nvfp4 | fp8|fp8-r2 | 136 vs 0 | +5.67 [4.79, 6.58] | [4.79, 6.54] | [4.30, 7.19] | yes |
| bf16-srv|nvfp4 | nvfp4|nvfp4-r2 | 136 vs 22 | +4.75 [3.83, 5.71] | [3.83, 5.62] | [3.42, 6.20] | yes |
| bf16-srv|nvfp4 | gguf-bf16|gguf-bf16-r2 | 136 vs 0 | +5.67 [4.79, 6.58] | [4.79, 6.54] | [4.30, 7.19] | yes |
| bf16-srv|nvfp4 | ref|ref-r2 | 136 vs 15 | +5.04 [4.12, 6.00] | [4.12, 5.88] | [3.63, 6.46] | yes |
| bf16-srv|nvfp4 | ref|ref-fla | 136 vs 14 | +5.08 [4.17, 6.00] | [4.21, 5.92] | [3.79, 6.67] | yes |
| bf16-srv|nvfp4 | ref|bf16-srv | 136 vs 13 | +5.12 [4.25, 6.04] | [4.21, 5.96] | [3.79, 6.58] | yes |
| bf16-srv|nvfp4 | ref|gguf-bf16 | 136 vs 16 | +5.00 [4.08, 5.92] | [4.08, 5.88] | [3.59, 6.42] | yes |
| gguf-bf16|q4_k_m | bf16-srv|bf16-srv-r2 | 84 vs 4 | +3.33 [2.62, 4.08] | [2.63, 4.08] | [2.21, 4.54] | yes |
| gguf-bf16|q4_k_m | fp8|fp8-r2 | 84 vs 0 | +3.50 [2.79, 4.29] | [2.79, 4.25] | [2.46, 4.71] | yes |
| gguf-bf16|q4_k_m | nvfp4|nvfp4-r2 | 84 vs 22 | +2.58 [1.79, 3.46] | [1.79, 3.42] | [1.30, 3.83] | yes |
| gguf-bf16|q4_k_m | gguf-bf16|gguf-bf16-r2 | 84 vs 0 | +3.50 [2.79, 4.29] | [2.79, 4.25] | [2.46, 4.71] | yes |
| gguf-bf16|q4_k_m | ref|ref-r2 | 84 vs 15 | +2.88 [2.17, 3.67] | [2.17, 3.67] | [1.75, 4.16] | yes |
| gguf-bf16|q4_k_m | ref|ref-fla | 84 vs 14 | +2.92 [2.21, 3.71] | [2.21, 3.67] | [1.83, 4.08] | yes |
| gguf-bf16|q4_k_m | ref|bf16-srv | 84 vs 13 | +2.96 [2.21, 3.75] | [2.21, 3.75] | [1.83, 4.17] | yes |
| gguf-bf16|q4_k_m | ref|gguf-bf16 | 84 vs 16 | +2.83 [2.12, 3.62] | [2.12, 3.58] | [1.71, 4.04] | yes |

Bonferroni intervals use the alpha/k tails of the same 2000 resamples, so the extreme tail rests on about three resamples; read them as approximate.

Every CI excludes 0 (lower bound > 0): True (stratified: True, Bonferroni: True). N2 supported: True.

### N3 engine change vs 8-bit quantization

Criterion: supported if each engine change at BF16 flips at least as many decisions as each 8-bit quantization inside one engine: point estimate of (engine - 8-bit) >= 0; contradicted for a combination when the 95% CI of the difference lies below 0.

| engine change | minus 8-bit within engine | flips | diff pp [95% CI] | strat [95% CI] | CI entirely below 0 |
|---|---|---|---|---|---|
| ref|bf16-srv | bf16-srv|fp8 | 13 vs 39 | -1.08 [-1.54, -0.62] | [-1.54, -0.62] | yes |
| ref|bf16-srv | gguf-bf16|q8_0 | 13 vs 25 | -0.50 [-0.92, -0.08] | [-0.92, -0.08] | yes |
| ref|gguf-bf16 | bf16-srv|fp8 | 16 vs 39 | -0.96 [-1.46, -0.50] | [-1.42, -0.50] | yes |
| ref|gguf-bf16 | gguf-bf16|q8_0 | 16 vs 25 | -0.37 [-0.79, 0.00] | [-0.75, 0.00] | no |

All four point estimates >= 0: False. Any engine change significantly smaller than an 8-bit effect: True. N3 supported: False.

### N4 kernel selection

Criterion: supported if ref-fla flips a nonzero number of decisions against ref AND that count lies within the range of the vLLM restart flip counts (the three measured vLLM restart pairs).

ref vs ref-fla: 14 flips of 2400 (0.58%); nonzero: True. vLLM restart flip counts: bf16-srv|bf16-srv-r2 = 4, fp8|fp8-r2 = 0, nvfp4|nvfp4-r2 = 22 (range 0 to 22). Within that range: True. Against the BF16 vLLM restart alone (4 flips) it is above.

| difference | flips | diff pp [95% CI] | excludes 0 |
|---|---|---|---|
| flip(ref|ref-fla) - flip(bf16-srv|bf16-srv-r2) | 14 vs 4 | +0.42 [0.08, 0.75] | yes |
| flip(ref|ref-fla) - flip(fp8|fp8-r2) | 14 vs 0 | +0.58 [0.29, 0.88] | yes |
| flip(ref|ref-fla) - flip(nvfp4|nvfp4-r2) | 14 vs 22 | -0.33 [-0.83, 0.13] | no |

N4 supported: True.

### Supplementary for H1 (not pre-registered)

Q8_0 vs BF16 inside llama.cpp minus the llama.cpp restart flip rate (N1's analogue for the GGUF 8-bit build): 25 vs 0 flips, diff +1.04 pp [0.67, 1.46], excludes 0: True.

### N5 llama.cpp BF16 restart

Criterion: supported if every record of gguf-bf16 and gguf-bf16-r2 is bit-identical.

gguf-bf16 vs gguf-bf16-r2: 2400 of 2400 records bit-identical (100.00%; basis: probs), 0 flips, max |dprob| 0, mean TVD 0. N5 supported: True.

## Run conditions of each config on the unit (from the result lines)

| config | engine | engine_version | batch_size | tasks | result lines read | lines outside the unit (ignored) |
|---|---|---|---|---|
| ref | transformers-native | transformers 5.16.1; gdn=transformers-torch-fallback | batch_size=1 | 12 | 31275 | 28875 |
| bf16-srv | vllm+clef_bench_vllm | vllm 0.29.0+5013de39.dev | batch_size=None | 12 | 31275 | 28875 |
| fp8 | vllm+clef_bench_vllm | vllm 0.29.0+5013de39.dev | batch_size=None | 12 | 31275 | 28875 |
| nvfp4 | vllm+clef_bench_vllm | vllm 0.29.0+5013de39.dev | batch_size=None | 12 | 31275 | 28875 |
| gguf-bf16 | llama.cpp | b11671-ba3c05f71 | batch_size=None | 12 | 31275 | 28875 |
| q8_0 | llama.cpp | b11671-ba3c05f71 | batch_size=None | 12 | 31275 | 28875 |
| q4_k_m | llama.cpp | b11671-ba3c05f71 | batch_size=None | 12 | 31275 | 28875 |
| ref-r2 | transformers-native | transformers 5.16.1; gdn=transformers-torch-fallback | batch_size=8 | 12 | 31275 | 28875 |
| ref-fla | transformers-native | transformers 5.16.1; gdn=fla-0.5.2 | batch_size=1 | 12 | 2400 | 0 |
| bf16-srv-r2 | vllm+clef_bench_vllm | vllm 0.29.0+5013de39.dev | batch_size=None | 12 | 2400 | 0 |
| fp8-r2 | vllm+clef_bench_vllm | vllm 0.29.0+5013de39.dev | batch_size=None | 12 | 2400 | 0 |
| nvfp4-r2 | vllm+clef_bench_vllm | vllm 0.29.0+5013de39.dev | batch_size=None | 12 | 2400 | 0 |
| gguf-bf16-r2 | llama.cpp | b11671-ba3c05f71 | batch_size=None | 12 | 2400 | 0 |

The `-r2` repeats and `ref-fla` hold exactly the first 200 records of each task (2,400 lines, none outside the unit); full-set configs hold the whole task and only the first 200 are used.

