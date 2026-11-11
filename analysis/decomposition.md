# Noise-floor decomposition: clef-flash

Comparison unit: first 200 records of each of 12 public tasks (2400 records, 2400 questions). Bootstrap: 2000 paired resamples over records, seed 20261015, percentile 95% CI; one index matrix is shared by every row, so differences are paired. Flip = top-1 option differs between the two configs (rates are shares of questions; one question per record on these tasks, so questions = records). TVD = mean total-variation distance of the option distributions. Bit-identical = all logits equal (logits present on both sides) or all stored probabilities equal (llama.cpp, which returns full-precision probabilities and no logits). max |dlogit| raw includes any constant shift of the logit scale, which softmax ignores (it dominates NVFP4 pairs); the centred value removes it. Exact = Clopper-Pearson interval (questions treated as independent), shown because the percentile bootstrap collapses to a point at 0 flips.

## Restart of the same vLLM flags (new server instance)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16-srv vs bf16-srv-r2 | vLLM restart, BF16 | 2400 | 0 | 0.00% [0.00, 0.00] | [0.00, 0.15] | 0.00009 [0.00007, 0.00012] | 0.0113 | 0.121 | 0.094 | 2249 (93.7%) [logits] |
| fp8 vs fp8-r2 | vLLM restart, FP8 | 2400 | 5 | 0.21% [0.04, 0.42] | [0.07, 0.49] | 0.00167 [0.00154, 0.00183] | 0.0509 | 0.609 | 0.239 | 360 (15.0%) [logits] |
| nvfp4 vs nvfp4-r2 | vLLM restart, NVFP4 | 2400 | 20 | 0.83% [0.50, 1.21] | [0.51, 1.28] | 0.00690 [0.00597, 0.00785] | 0.2932 | 2.875 | 1.232 | 1842 (76.8%) [logits] |

## Restart of llama.cpp (new server instance)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| gguf-bf16 vs gguf-bf16-r2 | llama.cpp restart, BF16 (N5) | 2400 | 0 | 0.00% [0.00, 0.00] | [0.00, 0.15] | 0.00000 [0.00000, 0.00000] | 0.0000 | - | - | 2400 (100.0%) [probs] |

## Native rerun at batch 8 (rerun + batch-size change, NOT a pure restart)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| ref vs ref-r2 | native rerun at batch 8 vs batch 1 (a rerun AND a batch-size change, not a pure restart) | 2400 | 8 | 0.33% [0.12, 0.58] | [0.14, 0.66] | 0.00266 [0.00249, 0.00282] | 0.0566 | 0.719 | 0.141 | 172 (7.2%) [logits] |

## Kernel selection (Gated DeltaNet: PyTorch fallback vs flash-linear-attention)

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| ref vs ref-fla | Gated DeltaNet kernel: PyTorch fallback vs flash-linear-attention (N4) | 2400 | 10 | 0.42% [0.21, 0.71] | [0.20, 0.76] | 0.00289 [0.00272, 0.00307] | 0.0450 | 0.859 | 0.257 | 164 (6.8%) [logits] |

## Engine change at BF16

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| ref vs bf16-srv | engine at BF16: transformers -> vLLM (N3) | 2400 | 8 | 0.33% [0.12, 0.58] | [0.14, 0.66] | 0.00308 [0.00289, 0.00327] | 0.0442 | 0.781 | 0.133 | 151 (6.3%) [logits] |
| ref vs gguf-bf16 | engine at BF16: transformers -> llama.cpp (N3) | 2400 | 6 | 0.25% [0.08, 0.46] | [0.09, 0.54] | 0.00264 [0.00247, 0.00280] | 0.0718 | - | - | 0 (0.0%) [probs] |

## Quantization inside one engine

| pair | what | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|---|
| bf16-srv vs fp8 | FP8 vs BF16 inside vLLM (N1) | 2400 | 51 | 2.12% [1.54, 2.71] | [1.59, 2.78] | 0.01409 [0.01315, 0.01507] | 0.2793 | 1.906 | 0.680 | 3 (0.1%) [logits] |
| bf16-srv vs nvfp4 | NVFP4 vs BF16 inside vLLM (N2) | 2400 | 139 | 5.79% [4.88, 6.71] | [4.89, 6.80] | 0.05161 [0.04862, 0.05480] | 0.7190 | 4.414 | 2.294 | 1 (0.0%) [logits] |
| gguf-bf16 vs q8_0 | Q8_0 vs BF16 inside llama.cpp (N3 8-bit) | 2400 | 21 | 0.88% [0.54, 1.29] | [0.54, 1.33] | 0.00559 [0.00519, 0.00602] | 0.1388 | - | - | 0 (0.0%) [probs] |
| gguf-bf16 vs q4_k_m | Q4_K_M vs BF16 inside llama.cpp (N2) | 2400 | 139 | 5.79% [4.92, 6.79] | [4.89, 6.80] | 0.04219 [0.03944, 0.04487] | 0.5915 | - | - | 0 (0.0%) [probs] |

## Every config against ref (for completeness)

| pair | n | flips | flip rate [95% CI] | flip rate exact 95% | mean TVD [95% CI] | max dprob | max dlogit raw | max dlogit centred | bit-identical records (share) [basis] |
|---|---|---|---|---|---|---|---|---|---|
| ref vs bf16-srv | 2400 | 8 | 0.33% [0.12, 0.58] | [0.14, 0.66] | 0.00308 [0.00289, 0.00327] | 0.0442 | 0.781 | 0.133 | 151 (6.3%) [logits] |
| ref vs fp8 | 2400 | 47 | 1.96% [1.42, 2.54] | [1.44, 2.60] | 0.01415 [0.01320, 0.01515] | 0.2757 | 2.031 | 0.668 | 6 (0.2%) [logits] |
| ref vs nvfp4 | 2400 | 143 | 5.96% [5.00, 6.92] | [5.04, 6.98] | 0.05168 [0.04863, 0.05485] | 0.7135 | 4.359 | 2.271 | 0 (0.0%) [logits] |
| ref vs gguf-bf16 | 2400 | 6 | 0.25% [0.08, 0.46] | [0.09, 0.54] | 0.00264 [0.00247, 0.00280] | 0.0718 | - | - | 0 (0.0%) [probs] |
| ref vs q8_0 | 2400 | 21 | 0.88% [0.54, 1.25] | [0.54, 1.33] | 0.00626 [0.00583, 0.00669] | 0.1445 | - | - | 0 (0.0%) [probs] |
| ref vs q4_k_m | 2400 | 135 | 5.62% [4.75, 6.62] | [4.74, 6.62] | 0.04221 [0.03949, 0.04489] | 0.6012 | - | - | 0 (0.0%) [probs] |
| ref vs ref-r2 | 2400 | 8 | 0.33% [0.12, 0.58] | [0.14, 0.66] | 0.00266 [0.00249, 0.00282] | 0.0566 | 0.719 | 0.141 | 172 (7.2%) [logits] |
| ref vs ref-fla | 2400 | 10 | 0.42% [0.21, 0.71] | [0.20, 0.76] | 0.00289 [0.00272, 0.00307] | 0.0450 | 0.859 | 0.257 | 164 (6.8%) [logits] |
| ref vs bf16-srv-r2 | 2400 | 8 | 0.33% [0.12, 0.58] | [0.14, 0.66] | 0.00307 [0.00288, 0.00326] | 0.0442 | 0.781 | 0.133 | 153 (6.4%) [logits] |
| ref vs fp8-r2 | 2400 | 48 | 2.00% [1.46, 2.58] | [1.48, 2.64] | 0.01418 [0.01324, 0.01521] | 0.2774 | 2.078 | 0.670 | 5 (0.2%) [logits] |
| ref vs nvfp4-r2 | 2400 | 151 | 6.29% [5.29, 7.29] | [5.35, 7.34] | 0.05217 [0.04908, 0.05543] | 0.7135 | 4.359 | 2.271 | 0 (0.0%) [logits] |
| ref vs gguf-bf16-r2 | 2400 | 6 | 0.25% [0.08, 0.46] | [0.09, 0.54] | 0.00264 [0.00247, 0.00280] | 0.0718 | - | - | 0 (0.0%) [probs] |

## Pre-registered tests (PLAN.md, N1-N5)

Each row is a paired-bootstrap interval of a DIFFERENCE of flip rates on the same records (percentage points). `strat` = same test with records resampled inside each task. Outcomes are reported as found.

### N1 FP8 inside restart noise

Criterion: supported if FP8's flip rate against bf16-srv is NOT significantly larger than the bf16-srv restart flip rate, i.e. the 95% CI of the difference does not lie above 0.

| difference | flips | diff pp [95% CI] | strat diff pp [95% CI] | excludes 0 |
|---|---|---|---|---|
| flip(bf16-srv|fp8) - flip(bf16-srv|bf16-srv-r2) | 51 vs 0 | +2.125 [1.542, 2.708] | [1.583, 2.709] | yes |

N1 supported: False.

Supplementary, not pre-registered (FP8 vs BF16 minus FP8 against its own restart (fp8 vs fp8-r2)): 51 vs 5 flips, diff +1.92 pp [1.33, 2.50], excludes 0: True.

### N2 NVFP4 and Q4_K_M exceed every non-quantization source

Criterion: supported if every quantization flip rate (NVFP4 vs bf16-srv, Q4_K_M vs gguf-bf16) exceeds every non-quantization source with a 95% CI of the difference that excludes 0 (lower bound > 0).

| quantization | minus source | flips | diff pp [95% CI] | strat [95% CI] | Bonferroni (k=16) [CI] | lower > 0 |
|---|---|---|---|---|---|---|
| bf16-srv|nvfp4 | bf16-srv|bf16-srv-r2 | 139 vs 0 | +5.79 [4.88, 6.71] | [4.92, 6.63] | [4.51, 7.24] | yes |
| bf16-srv|nvfp4 | fp8|fp8-r2 | 139 vs 5 | +5.58 [4.62, 6.50] | [4.71, 6.46] | [4.22, 7.00] | yes |
| bf16-srv|nvfp4 | nvfp4|nvfp4-r2 | 139 vs 20 | +4.96 [4.00, 5.96] | [4.04, 5.88] | [3.58, 6.29] | yes |
| bf16-srv|nvfp4 | gguf-bf16|gguf-bf16-r2 | 139 vs 0 | +5.79 [4.88, 6.71] | [4.92, 6.63] | [4.51, 7.24] | yes |
| bf16-srv|nvfp4 | ref|ref-r2 | 139 vs 8 | +5.46 [4.50, 6.42] | [4.58, 6.37] | [4.12, 6.95] | yes |
| bf16-srv|nvfp4 | ref|ref-fla | 139 vs 10 | +5.38 [4.46, 6.25] | [4.50, 6.25] | [4.09, 6.86] | yes |
| bf16-srv|nvfp4 | ref|bf16-srv | 139 vs 8 | +5.46 [4.54, 6.38] | [4.58, 6.33] | [4.09, 6.88] | yes |
| bf16-srv|nvfp4 | ref|gguf-bf16 | 139 vs 6 | +5.54 [4.62, 6.46] | [4.67, 6.46] | [4.17, 7.00] | yes |
| gguf-bf16|q4_k_m | bf16-srv|bf16-srv-r2 | 139 vs 0 | +5.79 [4.92, 6.79] | [4.92, 6.71] | [4.46, 7.25] | yes |
| gguf-bf16|q4_k_m | fp8|fp8-r2 | 139 vs 5 | +5.58 [4.71, 6.54] | [4.75, 6.50] | [4.34, 7.04] | yes |
| gguf-bf16|q4_k_m | nvfp4|nvfp4-r2 | 139 vs 20 | +4.96 [4.04, 5.96] | [4.04, 5.92] | [3.67, 6.42] | yes |
| gguf-bf16|q4_k_m | gguf-bf16|gguf-bf16-r2 | 139 vs 0 | +5.79 [4.92, 6.79] | [4.92, 6.71] | [4.46, 7.25] | yes |
| gguf-bf16|q4_k_m | ref|ref-r2 | 139 vs 8 | +5.46 [4.58, 6.50] | [4.58, 6.42] | [4.17, 6.91] | yes |
| gguf-bf16|q4_k_m | ref|ref-fla | 139 vs 10 | +5.38 [4.50, 6.38] | [4.54, 6.29] | [4.13, 6.86] | yes |
| gguf-bf16|q4_k_m | ref|bf16-srv | 139 vs 8 | +5.46 [4.58, 6.42] | [4.58, 6.38] | [4.17, 6.91] | yes |
| gguf-bf16|q4_k_m | ref|gguf-bf16 | 139 vs 6 | +5.54 [4.67, 6.54] | [4.67, 6.46] | [4.21, 7.00] | yes |

Bonferroni intervals use the alpha/k tails of the same 2000 resamples, so the extreme tail rests on about three resamples; read them as approximate.

Every CI excludes 0 (lower bound > 0): True (stratified: True, Bonferroni: True). N2 supported: True.

### N3 engine change vs 8-bit quantization

Criterion: supported if each engine change at BF16 flips at least as many decisions as each 8-bit quantization inside one engine: point estimate of (engine - 8-bit) >= 0; contradicted for a combination when the 95% CI of the difference lies below 0.

| engine change | minus 8-bit within engine | flips | diff pp [95% CI] | strat [95% CI] | CI entirely below 0 |
|---|---|---|---|---|---|
| ref|bf16-srv | bf16-srv|fp8 | 8 vs 51 | -1.79 [-2.33, -1.25] | [-2.38, -1.25] | yes |
| ref|bf16-srv | gguf-bf16|q8_0 | 8 vs 21 | -0.54 [-0.96, -0.17] | [-0.92, -0.17] | yes |
| ref|gguf-bf16 | bf16-srv|fp8 | 6 vs 51 | -1.88 [-2.42, -1.33] | [-2.46, -1.33] | yes |
| ref|gguf-bf16 | gguf-bf16|q8_0 | 6 vs 21 | -0.62 [-1.04, -0.25] | [-1.00, -0.25] | yes |

All four point estimates >= 0: False. Any engine change significantly smaller than an 8-bit effect: True. N3 supported: False.

### N4 kernel selection

Criterion: supported if ref-fla flips a nonzero number of decisions against ref AND that count lies within the range of the vLLM restart flip counts (the three measured vLLM restart pairs).

ref vs ref-fla: 10 flips of 2400 (0.42%); nonzero: True. vLLM restart flip counts: bf16-srv|bf16-srv-r2 = 0, fp8|fp8-r2 = 5, nvfp4|nvfp4-r2 = 20 (range 0 to 20). Within that range: True. Against the BF16 vLLM restart alone (0 flips) it is above.

| difference | flips | diff pp [95% CI] | excludes 0 |
|---|---|---|---|
| flip(ref|ref-fla) - flip(bf16-srv|bf16-srv-r2) | 10 vs 0 | +0.42 [0.21, 0.71] | yes |
| flip(ref|ref-fla) - flip(fp8|fp8-r2) | 10 vs 5 | +0.21 [-0.08, 0.54] | no |
| flip(ref|ref-fla) - flip(nvfp4|nvfp4-r2) | 10 vs 20 | -0.42 [-0.87, 0.00] | no |

N4 supported: True.

### Supplementary for H1 (not pre-registered)

Q8_0 vs BF16 inside llama.cpp minus the llama.cpp restart flip rate (N1's analogue for the GGUF 8-bit build): 21 vs 0 flips, diff +0.88 pp [0.54, 1.29], excludes 0: True.

### N5 llama.cpp BF16 restart

Criterion: supported if every record of gguf-bf16 and gguf-bf16-r2 is bit-identical.

gguf-bf16 vs gguf-bf16-r2: 2400 of 2400 records bit-identical (100.00%; basis: probs), 0 flips, max |dprob| 0, mean TVD 0. N5 supported: True.

## Run conditions of each config on the unit (from the result lines)

| config | engine | engine_version | batch_size | tasks | result lines read | lines outside the unit (ignored) |
|---|---|---|---|---|
| ref | transformers-native | transformers 5.16.1 | batch_size=1 | 6 | 31275 | 28875 |
| ref | transformers-native | transformers 5.16.1; gdn=transformers-torch-fallback | batch_size=1 | 6 | 31275 | 28875 |
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

