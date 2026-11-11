# Exploratory checks

Exploratory (not pre-registered). Public tasks only. Specified during review, after the results existed.

## Harm split against `ref`

Full set, 31275 questions.

| model | build | flips | worse | better | wrong to wrong | net |
|---|---|---|---|---|---|---|
| clef-flash | q8_0 | 183 | 83 | 82 | 18 | -1 |
| clef-flash | fp8 | 374 | 193 | 129 | 52 | -64 |
| clef-flash | q4_k_m | 1108 | 527 | 413 | 168 | -114 |
| clef-flash | nvfp4 | 1426 | 769 | 472 | 185 | -297 |
| clef | q8_0 | 264 | 126 | 104 | 34 | -22 |
| clef | fp8 | 520 | 241 | 208 | 71 | -33 |
| clef | q4_k_m | 1000 | 436 | 430 | 134 | -6 |
| clef | nvfp4 | 1502 | 762 | 560 | 180 | -202 |

2,400-record repeat unit, 2400 questions.

| model | build | flips | worse | better | wrong to wrong | net |
|---|---|---|---|---|---|---|
| clef-flash | q8_0 | 21 | 10 | 8 | 3 | -2 |
| clef-flash | fp8 | 47 | 24 | 16 | 7 | -8 |
| clef-flash | q4_k_m | 135 | 65 | 52 | 18 | -13 |
| clef-flash | nvfp4 | 143 | 78 | 49 | 16 | -29 |
| clef | q8_0 | 23 | 11 | 10 | 2 | -1 |
| clef | fp8 | 33 | 15 | 11 | 7 | -4 |
| clef | q4_k_m | 84 | 38 | 33 | 13 | -5 |
| clef | nvfp4 | 137 | 66 | 50 | 21 | -16 |

## Unit note: flip rate against `ref`, %

| model | build | full set, pooled | full set, tasks equal | repeat unit | equal / pooled | unit / pooled |
|---|---|---|---|---|---|---|
| clef-flash | q8_0 | 0.59 | 0.77 | 0.88 | 1.31 | 1.49 |
| clef-flash | fp8 | 1.2 | 1.63 | 1.96 | 1.36 | 1.63 |
| clef-flash | q4_k_m | 3.54 | 4.77 | 5.62 | 1.35 | 1.59 |
| clef-flash | nvfp4 | 4.56 | 6.17 | 5.96 | 1.35 | 1.31 |
| clef | q8_0 | 0.84 | 1.12 | 0.96 | 1.33 | 1.14 |
| clef | fp8 | 1.66 | 2.14 | 1.38 | 1.29 | 0.83 |
| clef | q4_k_m | 3.2 | 4.04 | 3.5 | 1.26 | 1.09 |
| clef | nvfp4 | 4.8 | 6.14 | 5.71 | 1.28 | 1.19 |

## Separation on the repeat unit: quantization minus non-quantization source, pp [95% CI]

**clef-flash**

| quantization | source | flips | difference | excludes 0 |
|---|---|---|---|---|
| bf16-srv|fp8 | bf16-srv|bf16-srv-r2 | 51 vs 0 | +2.12 [+1.54, +2.71] | yes |
| bf16-srv|fp8 | fp8|fp8-r2 | 51 vs 5 | +1.92 [+1.33, +2.50] | yes |
| bf16-srv|fp8 | nvfp4|nvfp4-r2 | 51 vs 20 | +1.29 [+0.67, +1.96] | yes |
| bf16-srv|fp8 | gguf-bf16|gguf-bf16-r2 | 51 vs 0 | +2.12 [+1.54, +2.71] | yes |
| bf16-srv|fp8 | ref|ref-r2 | 51 vs 8 | +1.79 [+1.25, +2.38] | yes |
| bf16-srv|fp8 | ref|ref-fla | 51 vs 10 | +1.71 [+1.08, +2.29] | yes |
| bf16-srv|fp8 | ref|bf16-srv | 51 vs 8 | +1.79 [+1.25, +2.33] | yes |
| bf16-srv|fp8 | ref|gguf-bf16 | 51 vs 6 | +1.88 [+1.33, +2.42] | yes |
| gguf-bf16|q8_0 | bf16-srv|bf16-srv-r2 | 21 vs 0 | +0.88 [+0.54, +1.29] | yes |
| gguf-bf16|q8_0 | fp8|fp8-r2 | 21 vs 5 | +0.67 [+0.29, +1.08] | yes |
| gguf-bf16|q8_0 | nvfp4|nvfp4-r2 | 21 vs 20 | +0.04 [-0.42, +0.54] | NO |
| gguf-bf16|q8_0 | gguf-bf16|gguf-bf16-r2 | 21 vs 0 | +0.88 [+0.54, +1.29] | yes |
| gguf-bf16|q8_0 | ref|ref-r2 | 21 vs 8 | +0.54 [+0.21, +0.92] | yes |
| gguf-bf16|q8_0 | ref|ref-fla | 21 vs 10 | +0.46 [+0.08, +0.88] | yes |
| gguf-bf16|q8_0 | ref|bf16-srv | 21 vs 8 | +0.54 [+0.17, +0.96] | yes |
| gguf-bf16|q8_0 | ref|gguf-bf16 | 21 vs 6 | +0.62 [+0.25, +1.04] | yes |
| bf16-srv|nvfp4 | bf16-srv|bf16-srv-r2 | 139 vs 0 | +5.79 [+4.88, +6.71] | yes |
| bf16-srv|nvfp4 | fp8|fp8-r2 | 139 vs 5 | +5.58 [+4.62, +6.50] | yes |
| bf16-srv|nvfp4 | nvfp4|nvfp4-r2 | 139 vs 20 | +4.96 [+4.00, +5.96] | yes |
| bf16-srv|nvfp4 | gguf-bf16|gguf-bf16-r2 | 139 vs 0 | +5.79 [+4.88, +6.71] | yes |
| bf16-srv|nvfp4 | ref|ref-r2 | 139 vs 8 | +5.46 [+4.50, +6.42] | yes |
| bf16-srv|nvfp4 | ref|ref-fla | 139 vs 10 | +5.38 [+4.46, +6.25] | yes |
| bf16-srv|nvfp4 | ref|bf16-srv | 139 vs 8 | +5.46 [+4.54, +6.38] | yes |
| bf16-srv|nvfp4 | ref|gguf-bf16 | 139 vs 6 | +5.54 [+4.62, +6.46] | yes |
| gguf-bf16|q4_k_m | bf16-srv|bf16-srv-r2 | 139 vs 0 | +5.79 [+4.92, +6.79] | yes |
| gguf-bf16|q4_k_m | fp8|fp8-r2 | 139 vs 5 | +5.58 [+4.71, +6.54] | yes |
| gguf-bf16|q4_k_m | nvfp4|nvfp4-r2 | 139 vs 20 | +4.96 [+4.04, +5.96] | yes |
| gguf-bf16|q4_k_m | gguf-bf16|gguf-bf16-r2 | 139 vs 0 | +5.79 [+4.92, +6.79] | yes |
| gguf-bf16|q4_k_m | ref|ref-r2 | 139 vs 8 | +5.46 [+4.58, +6.50] | yes |
| gguf-bf16|q4_k_m | ref|ref-fla | 139 vs 10 | +5.38 [+4.50, +6.38] | yes |
| gguf-bf16|q4_k_m | ref|bf16-srv | 139 vs 8 | +5.46 [+4.58, +6.42] | yes |
| gguf-bf16|q4_k_m | ref|gguf-bf16 | 139 vs 6 | +5.54 [+4.67, +6.54] | yes |

**clef**

| quantization | source | flips | difference | excludes 0 |
|---|---|---|---|---|
| bf16-srv|fp8 | bf16-srv|bf16-srv-r2 | 39 vs 4 | +1.46 [+0.96, +1.96] | yes |
| bf16-srv|fp8 | fp8|fp8-r2 | 39 vs 0 | +1.62 [+1.12, +2.12] | yes |
| bf16-srv|fp8 | nvfp4|nvfp4-r2 | 39 vs 22 | +0.71 [+0.08, +1.29] | yes |
| bf16-srv|fp8 | gguf-bf16|gguf-bf16-r2 | 39 vs 0 | +1.62 [+1.12, +2.12] | yes |
| bf16-srv|fp8 | ref|ref-r2 | 39 vs 15 | +1.00 [+0.50, +1.50] | yes |
| bf16-srv|fp8 | ref|ref-fla | 39 vs 14 | +1.04 [+0.54, +1.58] | yes |
| bf16-srv|fp8 | ref|bf16-srv | 39 vs 13 | +1.08 [+0.62, +1.54] | yes |
| bf16-srv|fp8 | ref|gguf-bf16 | 39 vs 16 | +0.96 [+0.50, +1.46] | yes |
| gguf-bf16|q8_0 | bf16-srv|bf16-srv-r2 | 25 vs 4 | +0.87 [+0.50, +1.29] | yes |
| gguf-bf16|q8_0 | fp8|fp8-r2 | 25 vs 0 | +1.04 [+0.67, +1.46] | yes |
| gguf-bf16|q8_0 | nvfp4|nvfp4-r2 | 25 vs 22 | +0.12 [-0.42, +0.67] | NO |
| gguf-bf16|q8_0 | gguf-bf16|gguf-bf16-r2 | 25 vs 0 | +1.04 [+0.67, +1.46] | yes |
| gguf-bf16|q8_0 | ref|ref-r2 | 25 vs 15 | +0.42 [+0.04, +0.83] | yes |
| gguf-bf16|q8_0 | ref|ref-fla | 25 vs 14 | +0.46 [+0.08, +0.88] | yes |
| gguf-bf16|q8_0 | ref|bf16-srv | 25 vs 13 | +0.50 [+0.08, +0.92] | yes |
| gguf-bf16|q8_0 | ref|gguf-bf16 | 25 vs 16 | +0.37 [+0.00, +0.79] | NO |
| bf16-srv|nvfp4 | bf16-srv|bf16-srv-r2 | 136 vs 4 | +5.50 [+4.62, +6.42] | yes |
| bf16-srv|nvfp4 | fp8|fp8-r2 | 136 vs 0 | +5.67 [+4.79, +6.58] | yes |
| bf16-srv|nvfp4 | nvfp4|nvfp4-r2 | 136 vs 22 | +4.75 [+3.83, +5.71] | yes |
| bf16-srv|nvfp4 | gguf-bf16|gguf-bf16-r2 | 136 vs 0 | +5.67 [+4.79, +6.58] | yes |
| bf16-srv|nvfp4 | ref|ref-r2 | 136 vs 15 | +5.04 [+4.12, +6.00] | yes |
| bf16-srv|nvfp4 | ref|ref-fla | 136 vs 14 | +5.08 [+4.17, +6.00] | yes |
| bf16-srv|nvfp4 | ref|bf16-srv | 136 vs 13 | +5.12 [+4.25, +6.04] | yes |
| bf16-srv|nvfp4 | ref|gguf-bf16 | 136 vs 16 | +5.00 [+4.08, +5.92] | yes |
| gguf-bf16|q4_k_m | bf16-srv|bf16-srv-r2 | 84 vs 4 | +3.33 [+2.62, +4.08] | yes |
| gguf-bf16|q4_k_m | fp8|fp8-r2 | 84 vs 0 | +3.50 [+2.79, +4.29] | yes |
| gguf-bf16|q4_k_m | nvfp4|nvfp4-r2 | 84 vs 22 | +2.58 [+1.79, +3.46] | yes |
| gguf-bf16|q4_k_m | gguf-bf16|gguf-bf16-r2 | 84 vs 0 | +3.50 [+2.79, +4.29] | yes |
| gguf-bf16|q4_k_m | ref|ref-r2 | 84 vs 15 | +2.88 [+2.17, +3.67] | yes |
| gguf-bf16|q4_k_m | ref|ref-fla | 84 vs 14 | +2.92 [+2.21, +3.71] | yes |
| gguf-bf16|q4_k_m | ref|bf16-srv | 84 vs 13 | +2.96 [+2.21, +3.75] | yes |
| gguf-bf16|q4_k_m | ref|gguf-bf16 | 84 vs 16 | +2.83 [+2.12, +3.62] | yes |

## Effect-to-floor ratios

| unit | model | floor used | q8_0 | fp8 | q4_k_m | nvfp4 |
|---|---|---|---|---|---|---|
| unit | clef-flash | 20 (nvfp4|nvfp4-r2) | 1.05 | 2.55 | 6.95 | 6.95 |
| full | clef-flash | 75 (ref|bf16-srv) | 2.44 | 4.99 | 14.77 | 19.01 |
| unit | clef | 22 (nvfp4|nvfp4-r2) | 1.14 | 1.77 | 3.82 | 6.18 |
| full | clef | 156 (ref|gguf-bf16) | 1.69 | 3.33 | 6.41 | 9.63 |
