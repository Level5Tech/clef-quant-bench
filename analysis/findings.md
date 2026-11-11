# Findings: Clef quantization benchmark

Every number below is read from a JSON file in this directory by `bench/findings.py` and carries a citation `[ALIAS key.path]`; the verbatim hypothesis text is parsed from `PLAN.md`. Hypotheses are fixed: outcomes are reported as found.

Sources: `PUB` = `analysis/flash_public/summary.json`; `PUBALL` = `analysis/all_public/summary.json`; `DEC` = `analysis/decomposition.json`; `HYP` = `analysis/hypotheses.json`; `PRIV` = `analysis/private_aggregates.json`; `SPD` = `analysis/speed.json`; `MAN` = `data/manifest.json`; `PRIVA` = `analysis/flash_private/private_summary.json` (analyzer output for the private sets). Private sets appear as aggregates only.

Scope: Clef-flash (9B) has results for 13 configs [SPD configs.clef-flash]; Clef (27B) for 13 [SPD configs.clef]. Clef (27B) covers all 12 public tasks with ref, bf16-srv, fp8, gguf-bf16, nvfp4, q4_k_m, q8_0, and only 2,400 public questions with ref-fla; its private-set pairs against ref are bf16-srv, fp8, gguf-bf16, nvfp4, q4_k_m, q8_0, ref-r2 [PRIV label_free_big]. Comparison unit for the noise-floor tests: the first 200 records of each public task (2,400 questions); full-set numbers use all 31,275 public questions. 'pp' = percentage points; CIs are 95%.

## Status at a glance

| id | status |
|---|---|
| H1 | not supported |
| H2 | mixed |
| H3 | supported |
| H4 | not supported |
| H5 | mixed |
| H6 | mixed |
| H7 | not supported |
| H8 | not evaluable |
| H9 | supported |
| H10 | mixed |
| H11 | supported |
| N1 | not supported |
| N2 | supported |
| N3 | not supported |
| N4 | supported |
| N5 | supported |

## Hypotheses H1-H11

### H1

> H1 8-bit is indistinguishable from BF16 within the noise floor.

- fp8 against ref, pooled over 12 tasks (31,275 questions): accuracy change -0.20 pp [-0.32, -0.09] [PUB models.clef-flash.configs.fp8.pooled.vs_ref.delta_accuracy.delta,ci95]; top-1 agreement 98.80% [PUB models.clef-flash.configs.fp8.pooled.vs_ref.agreement] (374 [PUB models.clef-flash.configs.fp8.pooled.vs_ref.n_flips] flips).
- q8_0 against ref, pooled over 12 tasks (31,275 questions): accuracy change -0.00 pp [-0.08, +0.08] [PUB models.clef-flash.configs.q8_0.pooled.vs_ref.delta_accuracy.delta,ci95]; top-1 agreement 99.41% [PUB models.clef-flash.configs.q8_0.pooled.vs_ref.agreement] (183 [PUB models.clef-flash.configs.q8_0.pooled.vs_ref.n_flips] flips).
- Measured full-set floor, ref vs ref-r2 (rerun at batch 8): agreement 99.79% [PUB models.clef-flash.noise_floor.ref.ref-r2.pooled.agreement], 66 [PUB models.clef-flash.noise_floor.ref.ref-r2.pooled.n_flips] flips.
- Same records, first 200 per task (n = 2,400): FP8 vs bf16-srv flips 51 [DEC pairs.bf16-srv|fp8.flips] (2.12% [1.54, 2.71] [DEC pairs.bf16-srv|fp8.flip_rate,flip_ci95]); vLLM BF16 restart 0 [DEC pairs.bf16-srv|bf16-srv-r2.flips] flips, FP8 restart 5 [DEC pairs.fp8|fp8-r2.flips] flips.
- Q8_0 vs gguf-bf16 flips 21 [DEC pairs.gguf-bf16|q8_0.flips] (0.88% [0.54, 1.29] [DEC pairs.gguf-bf16|q8_0.flip_rate,flip_ci95]) against the llama.cpp restart's 0 [DEC pairs.gguf-bf16|gguf-bf16-r2.flips]; difference +0.88 pp [+0.54, +1.29] [DEC tests.H1_supplementary_not_preregistered.diff,ci95] (not pre-registered).
- fp8 against its own engine's BF16, full set: accuracy -0.19 pp [-0.31, -0.08] [HYP calibration.within_engine.bf16-srv|fp8.accuracy.delta,ci95], ECE +0.00158 [+0.00047, +0.00282] [HYP calibration.within_engine.bf16-srv|fp8.ece.delta,ci95], Brier +0.00174 [+0.00121, +0.00226] [HYP calibration.within_engine.bf16-srv|fp8.brier.delta,ci95].
- q8_0 against its own engine's BF16, full set: accuracy +0.00 pp [-0.07, +0.08] [HYP calibration.within_engine.gguf-bf16|q8_0.accuracy.delta,ci95], ECE -0.00008 [-0.00089, +0.00079] [HYP calibration.within_engine.gguf-bf16|q8_0.ece.delta,ci95], Brier +0.00024 [+0.00003, +0.00043] [HYP calibration.within_engine.gguf-bf16|q8_0.brier.delta,ci95].

**Status: not supported.** FP8: accuracy CI excludes 0 and flips exceed the restart floor (51 vs 0). Q8_0: accuracy CI contains 0, but it flips 0.88% of decisions against a llama.cpp restart floor of 0.00%. Neither is indistinguishable within the measured floor.

Rule: an 8-bit config is indistinguishable from BF16 only if its full-set accuracy-change CI contains 0 AND the CI of its flip rate minus the matching restart floor (200-record unit) contains 0; supported if both 8-bit configs pass, mixed if one does, not supported if neither does.

### H2

> H2 4-bit loses accuracy, more on Clef-flash than Clef in absolute flips.

- nvfp4 against ref (Clef-flash, 31,275 questions): accuracy change -0.95 pp [-1.17, -0.71] [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.delta_accuracy.delta,ci95]; 1,426 [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.n_flips] top-1 flips (4.56% of questions) [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.agreement].
- q4_k_m against ref (Clef-flash, 31,275 questions): accuracy change -0.36 pp [-0.56, -0.18] [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.delta_accuracy.delta,ci95]; 1,108 [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.n_flips] top-1 flips (3.54% of questions) [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.agreement].
- nvfp4 against ref (Clef (27B), 31,275 questions, analyzer over both models): accuracy change -0.65 pp [-0.87, -0.42] [PUBALL models.clef.configs.nvfp4.pooled.vs_ref.delta_accuracy.delta,ci95]; 1,502 [PUBALL models.clef.configs.nvfp4.pooled.vs_ref.n_flips] top-1 flips (4.80% of questions) [PUBALL models.clef.configs.nvfp4.pooled.vs_ref.agreement]; Clef-flash, same build: 1,426 [PUBALL models.clef-flash.configs.nvfp4.pooled.vs_ref.n_flips] flips (4.56%) [PUBALL models.clef-flash.configs.nvfp4.pooled.vs_ref.agreement].
- q4_k_m against ref (Clef (27B), 31,275 questions, analyzer over both models): accuracy change -0.02 pp [-0.20, +0.16] [PUBALL models.clef.configs.q4_k_m.pooled.vs_ref.delta_accuracy.delta,ci95]; 1,000 [PUBALL models.clef.configs.q4_k_m.pooled.vs_ref.n_flips] top-1 flips (3.20% of questions) [PUBALL models.clef.configs.q4_k_m.pooled.vs_ref.agreement]; Clef-flash, same build: 1,108 [PUBALL models.clef-flash.configs.q4_k_m.pooled.vs_ref.n_flips] flips (3.54%) [PUBALL models.clef-flash.configs.q4_k_m.pooled.vs_ref.agreement].
- Private sets, pooled over R1, R2, R4a and R4b, nvfp4, flips against each model's own ref: Clef (27B) 1,195 of 27,738 (4.31%; sum over [PRIV label_free_big.<set>.pooled.pairs.ref|nvfp4.flips]) vs Clef-flash 2,168 of 27,738 (7.82%; sum over [PRIV label_free.<set>.pooled.pairs.ref|nvfp4.flips]).
- Private sets, pooled over R1, R2, R4a and R4b, q4_k_m, flips against each model's own ref: Clef (27B) 917 of 27,738 (3.31%; sum over [PRIV label_free_big.<set>.pooled.pairs.ref|q4_k_m.flips]) vs Clef-flash 2,425 of 27,738 (8.74%; sum over [PRIV label_free.<set>.pooled.pairs.ref|q4_k_m.flips]).

**Status: mixed.** First clause (4-bit loses accuracy; CI entirely below 0 against the model's own ref): holds for Clef-flash NVFP4, Clef-flash Q4_K_M, Clef (27B) NVFP4; fails for Clef (27B) Q4_K_M (CI upper bound +0.16 pp). Second clause (fewer top-1 flips on Clef (27B) than on Clef-flash; counts are 27B vs Clef-flash): holds for Q4_K_M (1,000 vs 1,108); fails for NVFP4 (1,502 vs 1,426). Some checks hold and some fail, so the hypothesis holds for some builds and not for others.

Rule: supported only if the first clause holds and the 27B 4-bit flip count on the same records is smaller than Clef-flash's. Reading added 2026-10-23, after the 27B results were seen: the first clause is checked for each 4-bit build of both models (accuracy-change CI entirely below 0, each model against its own BF16 ref, full public set); the second compares, per 4-bit build, the two models' top-1 flips against their own ref on the same 31,275 public questions. Supported if every check holds, not supported if no 4-bit build has fewer 27B flips or no build loses accuracy, mixed otherwise. Private flips are reported beside the rule and are not part of it.

### H3

> H3 flips concentrate where ref confidence is low.

- nvfp4: 85.9% [HYP confidence.per_config.nvfp4.share_of_flips_with_ref_conf_below_0_7] of its 1,426 [HYP confidence.per_config.nvfp4.flips] flips have ref confidence below 0.7, where only 16.3% [HYP confidence.per_config.nvfp4.share_of_items_with_ref_conf_below_0_7] of questions sit; flip rate below 0.7 is 24.1% [HYP confidence.per_config.nvfp4.flip_rate_ref_conf_below_0_7] vs 0.77% [HYP confidence.per_config.nvfp4.flip_rate_ref_conf_at_least_0_7] at or above it; highest ref confidence of any flip 0.969 [HYP confidence.per_config.nvfp4.max_ref_confidence_among_flips].
- q4_k_m: 90.2% [HYP confidence.per_config.q4_k_m.share_of_flips_with_ref_conf_below_0_7] of its 1,108 [HYP confidence.per_config.q4_k_m.flips] flips have ref confidence below 0.7, where only 16.3% [HYP confidence.per_config.q4_k_m.share_of_items_with_ref_conf_below_0_7] of questions sit; flip rate below 0.7 is 19.6% [HYP confidence.per_config.q4_k_m.flip_rate_ref_conf_below_0_7] vs 0.42% [HYP confidence.per_config.q4_k_m.flip_rate_ref_conf_at_least_0_7] at or above it; highest ref confidence of any flip 0.941 [HYP confidence.per_config.q4_k_m.max_ref_confidence_among_flips].
- fp8: 98.1% [HYP confidence.per_config.fp8.share_of_flips_with_ref_conf_below_0_7] of its 374 [HYP confidence.per_config.fp8.flips] flips have ref confidence below 0.7, where only 16.3% [HYP confidence.per_config.fp8.share_of_items_with_ref_conf_below_0_7] of questions sit; flip rate below 0.7 is 7.2% [HYP confidence.per_config.fp8.flip_rate_ref_conf_below_0_7] vs 0.03% [HYP confidence.per_config.fp8.flip_rate_ref_conf_at_least_0_7] at or above it; highest ref confidence of any flip 0.867 [HYP confidence.per_config.fp8.max_ref_confidence_among_flips].
- q8_0: 100.0% [HYP confidence.per_config.q8_0.share_of_flips_with_ref_conf_below_0_7] of its 183 [HYP confidence.per_config.q8_0.flips] flips have ref confidence below 0.7, where only 16.3% [HYP confidence.per_config.q8_0.share_of_items_with_ref_conf_below_0_7] of questions sit; flip rate below 0.7 is 3.6% [HYP confidence.per_config.q8_0.flip_rate_ref_conf_below_0_7] vs 0.00% [HYP confidence.per_config.q8_0.flip_rate_ref_conf_at_least_0_7] at or above it; highest ref confidence of any flip 0.663 [HYP confidence.per_config.q8_0.max_ref_confidence_among_flips].
- By bucket (pooled, NVFP4): flip rate 41.6% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.flip_by_ref_confidence.0.flip_rate] below 0.5, 18.5% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.flip_by_ref_confidence.1.flip_rate] in [0.5, 0.7), 0.06% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.flip_by_ref_confidence.3.flip_rate] in [0.9, 0.99), 0.0% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.flip_by_ref_confidence.4.flip_rate] at or above 0.99 (n = 81).

**Status: supported.** 86% to 100% of flips sit below 0.7. It is not a clean 'never above X' rule for 4-bit: NVFP4 still flips 11 of 18,125 questions with ref confidence at or above 0.9, while the 8-bit llama.cpp build's highest flipping confidence is 0.66 and FP8's 0.87.

Rule: supported if, for every quantized config, more than 80% of its flips fall at ref confidence below 0.7 (a bucket holding about 16% of questions).

### H4

> H4 calibration (ECE/Brier) degrades before accuracy does.

- bf16-srv|fp8 (quantized vs BF16 in the same engine, full set): accuracy -0.19 pp [-0.31, -0.08] [HYP calibration.within_engine.bf16-srv|fp8.accuracy.delta,ci95]; ECE +0.00158 [+0.00047, +0.00282] [HYP calibration.within_engine.bf16-srv|fp8.ece.delta,ci95]; Brier +0.00174 [+0.00121, +0.00226] [HYP calibration.within_engine.bf16-srv|fp8.brier.delta,ci95].
- gguf-bf16|q8_0 (quantized vs BF16 in the same engine, full set): accuracy +0.00 pp [-0.07, +0.08] [HYP calibration.within_engine.gguf-bf16|q8_0.accuracy.delta,ci95]; ECE -0.00008 [-0.00089, +0.00079] [HYP calibration.within_engine.gguf-bf16|q8_0.ece.delta,ci95]; Brier +0.00024 [+0.00003, +0.00043] [HYP calibration.within_engine.gguf-bf16|q8_0.brier.delta,ci95].
- bf16-srv|nvfp4 (quantized vs BF16 in the same engine, full set): accuracy -0.94 pp [-1.15, -0.71] [HYP calibration.within_engine.bf16-srv|nvfp4.accuracy.delta,ci95]; ECE +0.00116 [-0.00123, +0.00343] [HYP calibration.within_engine.bf16-srv|nvfp4.ece.delta,ci95]; Brier +0.01062 [+0.00894, +0.01215] [HYP calibration.within_engine.bf16-srv|nvfp4.brier.delta,ci95].
- gguf-bf16|q4_k_m (quantized vs BF16 in the same engine, full set): accuracy -0.36 pp [-0.55, -0.18] [HYP calibration.within_engine.gguf-bf16|q4_k_m.accuracy.delta,ci95]; ECE -0.00157 [-0.00345, +0.00046] [HYP calibration.within_engine.gguf-bf16|q4_k_m.ece.delta,ci95]; Brier +0.00434 [+0.00298, +0.00561] [HYP calibration.within_engine.gguf-bf16|q4_k_m.brier.delta,ci95].
- Non-quantization reference point, bf16-srv against ref: accuracy -0.01 pp [-0.06, +0.04] [HYP calibration.per_config.bf16-srv.delta_vs_ref.accuracy.delta,ci95]; ECE -0.00007 [-0.00060, +0.00056] [HYP calibration.per_config.bf16-srv.delta_vs_ref.ece.delta,ci95]; Brier +0.00024 [+0.00012, +0.00039] [HYP calibration.per_config.bf16-srv.delta_vs_ref.brier.delta,ci95].
- Non-quantization reference point, gguf-bf16 against ref: accuracy -0.01 pp [-0.06, +0.04] [HYP calibration.per_config.gguf-bf16.delta_vs_ref.accuracy.delta,ci95]; ECE +0.00037 [-0.00020, +0.00092] [HYP calibration.per_config.gguf-bf16.delta_vs_ref.ece.delta,ci95]; Brier +0.00029 [+0.00019, +0.00041] [HYP calibration.per_config.gguf-bf16.delta_vs_ref.brier.delta,ci95].
- Non-quantization reference point, ref-r2 against ref: accuracy +0.02 pp [-0.03, +0.06] [HYP calibration.per_config.ref-r2.delta_vs_ref.accuracy.delta,ci95]; ECE -0.00041 [-0.00082, +0.00058] [HYP calibration.per_config.ref-r2.delta_vs_ref.ece.delta,ci95]; Brier +0.00009 [+0.00002, +0.00017] [HYP calibration.per_config.ref-r2.delta_vs_ref.brier.delta,ci95].

**Status: not supported.** No build does. ECE does not move without accuracy moving (FP8's ECE rise comes with a significant accuracy loss; the other three ECE intervals contain 0). Q8_0's Brier rise (+0.00024) is significant but no larger than the Brier change from an engine change or a rerun alone (+0.00029), so it is not attributed to quantization.

Rule: calibration degrades before accuracy if some quantized build (against its own engine's BF16) shows a significant ECE increase, or a significant Brier increase larger than the largest non-quantization Brier change (+0.00029), while its accuracy-change CI still contains 0.

### H5

> H5 out-of-scope detection degrades more than in-scope classification.

- CLINC150 (n = 5,000, of which 909 gold out-of-scope): reference out-of-scope recall 92.63% [HYP oos.per_config.ref.oos_recall], in-scope accuracy 99.19% [HYP oos.per_config.ref.in_scope_acc].
- fp8: out-of-scope recall -0.66 pp [-1.33, -0.11] [HYP oos.per_config.fp8.delta_vs_ref.oos_recall.delta,ci95]; in-scope accuracy -0.02 pp [-0.07, +0.00] [HYP oos.per_config.fp8.delta_vs_ref.in_scope_acc.delta,ci95]; difference of the two changes -0.64 pp [-1.30, -0.07] [HYP oos.per_config.fp8.delta_vs_ref.oos_recall_minus_in_scope_acc.delta,ci95]; flip rate on out-of-scope items 0.88% [HYP oos.per_config.fp8.flip_rate_oos_items] vs in-scope items 0.02% [HYP oos.per_config.fp8.flip_rate_in_scope_items].
- q8_0: out-of-scope recall +0.22 pp [+0.00, +0.55] [HYP oos.per_config.q8_0.delta_vs_ref.oos_recall.delta,ci95]; in-scope accuracy -0.02 pp [-0.07, +0.00] [HYP oos.per_config.q8_0.delta_vs_ref.in_scope_acc.delta,ci95]; difference of the two changes +0.24 pp [+0.00, +0.59] [HYP oos.per_config.q8_0.delta_vs_ref.oos_recall_minus_in_scope_acc.delta,ci95]; flip rate on out-of-scope items 0.22% [HYP oos.per_config.q8_0.flip_rate_oos_items] vs in-scope items 0.02% [HYP oos.per_config.q8_0.flip_rate_in_scope_items].
- nvfp4: out-of-scope recall +0.33 pp [-0.68, +1.40] [HYP oos.per_config.nvfp4.delta_vs_ref.oos_recall.delta,ci95]; in-scope accuracy +0.00 pp [-0.12, +0.12] [HYP oos.per_config.nvfp4.delta_vs_ref.in_scope_acc.delta,ci95]; difference of the two changes +0.33 pp [-0.70, +1.41] [HYP oos.per_config.nvfp4.delta_vs_ref.oos_recall_minus_in_scope_acc.delta,ci95]; flip rate on out-of-scope items 2.64% [HYP oos.per_config.nvfp4.flip_rate_oos_items] vs in-scope items 0.20% [HYP oos.per_config.nvfp4.flip_rate_in_scope_items].
- q4_k_m: out-of-scope recall +0.33 pp [-0.56, +1.24] [HYP oos.per_config.q4_k_m.delta_vs_ref.oos_recall.delta,ci95]; in-scope accuracy -0.02 pp [-0.15, +0.10] [HYP oos.per_config.q4_k_m.delta_vs_ref.in_scope_acc.delta,ci95]; difference of the two changes +0.35 pp [-0.56, +1.27] [HYP oos.per_config.q4_k_m.delta_vs_ref.oos_recall_minus_in_scope_acc.delta,ci95]; flip rate on out-of-scope items 1.98% [HYP oos.per_config.q4_k_m.flip_rate_oos_items] vs in-scope items 0.17% [HYP oos.per_config.q4_k_m.flip_rate_in_scope_items].

**Status: mixed.** Only fp8 shows out-of-scope detection degrading more than in-scope classification. Both 4-bit builds keep out-of-scope recall (changes are small and their CIs contain 0) even though out-of-scope items flip 12-14 times as often as in-scope items; the flips go both ways.

Rule: supported if, for every quantized config, (change in out-of-scope recall) minus (change in in-scope accuracy) has a 95% CI entirely below 0; not supported if none does; mixed otherwise.

### H6

> H6 drift grows with input length, so real-world data (longer, messier) shows more drift than benchmarks.

- Public, fp8: flip rate against ref Q1 1.02% -> Q4 1.36% (quartiles of input tokens inside each task); Q4 minus Q1 +0.35 pp [+0.02, +0.69] [HYP length.per_config.fp8.q4_minus_q1.delta,ci95].
- Public, q8_0: flip rate against ref Q1 0.46% -> Q4 0.58% (quartiles of input tokens inside each task); Q4 minus Q1 +0.12 pp [-0.11, +0.34] [HYP length.per_config.q8_0.q4_minus_q1.delta,ci95].
- Public, nvfp4: flip rate against ref Q1 3.67% -> Q4 4.87% (quartiles of input tokens inside each task); Q4 minus Q1 +1.20 pp [+0.56, +1.83] [HYP length.per_config.nvfp4.q4_minus_q1.delta,ci95].
- Public, q4_k_m: flip rate against ref Q1 3.02% -> Q4 4.17% (quartiles of input tokens inside each task); Q4 minus Q1 +1.15 pp [+0.56, +1.74] [HYP length.per_config.q4_k_m.q4_minus_q1.delta,ci95].
- Public, bf16-srv: flip rate against ref Q1 0.20% -> Q4 0.23% (quartiles of input tokens inside each task); Q4 minus Q1 +0.03 pp [-0.11, +0.18] [HYP length.per_config.bf16-srv.q4_minus_q1.delta,ci95].
- Public, ref-r2: flip rate against ref Q1 0.15% -> Q4 0.27% (quartiles of input tokens inside each task); Q4 minus Q1 +0.12 pp [-0.02, +0.26] [HYP length.per_config.ref-r2.q4_minus_q1.delta,ci95].
- Private sets, nvfp4: of 10 (set, question) cells, Q4 flips significantly more than Q1 in 1, significantly less in 1, no significant difference in 8 [PRIV label_free.<set>.<question>.pairs.ref|nvfp4.flip_q4_minus_q1].
- Private sets, q4_k_m: of 10 (set, question) cells, Q4 flips significantly more than Q1 in 1, significantly less in 1, no significant difference in 8 [PRIV label_free.<set>.<question>.pairs.ref|q4_k_m.flip_q4_minus_q1].
- Flip rate against ref, fp8: private sets pooled 2.18% (606 of 27,738; sum over [PRIV label_free.<set>.pooled.pairs.ref|fp8.flips]) vs public pooled 1.20% (374 of 31,275) [PUB models.clef-flash.configs.fp8.pooled.vs_ref.n_flips].
- Flip rate against ref, q8_0: private sets pooled 0.90% (249 of 27,738; sum over [PRIV label_free.<set>.pooled.pairs.ref|q8_0.flips]) vs public pooled 0.59% (183 of 31,275) [PUB models.clef-flash.configs.q8_0.pooled.vs_ref.n_flips].
- Flip rate against ref, nvfp4: private sets pooled 7.82% (2,168 of 27,738; sum over [PRIV label_free.<set>.pooled.pairs.ref|nvfp4.flips]) vs public pooled 4.56% (1,426 of 31,275) [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.n_flips].
- Flip rate against ref, q4_k_m: private sets pooled 8.74% (2,425 of 27,738; sum over [PRIV label_free.<set>.pooled.pairs.ref|q4_k_m.flips]) vs public pooled 3.54% (1,108 of 31,275) [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.n_flips].
- Confidence mix: 48.1% of private questions have ref confidence below 0.7 against 16.3% [HYP confidence.per_config.nvfp4.share_of_items_with_ref_conf_below_0_7] of public questions, and H3 puts the flips there [PRIV label_free.<set>.pooled.pairs.ref|nvfp4.flip_by_ref_confidence].

**Status: mixed.** Clause 1 (CI of Q4 minus Q1 above 0) holds on the public sets for fp8, nvfp4, q4_k_m and not for q8_0, bf16-srv, ref-r2; in the private sets only 1 of 10 (NVFP4) and 1 of 10 (Q4_K_M) cells show it. Clause 2 holds on pooled rates for 4 of 4 builds; the private sets also hold far more low-confidence questions (48% vs 16%), where flips concentrate (H3), while length alone does not explain the private gap.

Rule: clause 1 (drift grows with input length): Q4 minus Q1 flip-rate CI above 0 for both 4-bit builds on the public sets AND in most private (set, question) cells; clause 2 (real-world data drifts more): private pooled flip rate above the public pooled flip rate for all four quantized builds. supported if both hold, not supported if neither, mixed otherwise.

### H7

> H7 engine change (transformers -> vLLM / llama.cpp) moves decisions as much as 8-bit quantization.

- Full set, bf16-srv against ref: 75 [PUB models.clef-flash.configs.bf16-srv.pooled.vs_ref.n_flips] flips (0.24%).
- Full set, gguf-bf16 against ref: 67 [PUB models.clef-flash.configs.gguf-bf16.pooled.vs_ref.n_flips] flips (0.21%).
- Full set, fp8 against ref: 374 [PUB models.clef-flash.configs.fp8.pooled.vs_ref.n_flips] flips (1.20%).
- Full set, q8_0 against ref: 183 [PUB models.clef-flash.configs.q8_0.pooled.vs_ref.n_flips] flips (0.59%).
- 200-record unit, flip(ref|bf16-srv) minus flip(bf16-srv|fp8): 8 vs 51 flips, -1.79 pp [-2.33, -1.25] [DEC tests.N3.rows.0.diff,ci95].
- 200-record unit, flip(ref|bf16-srv) minus flip(gguf-bf16|q8_0): 8 vs 21 flips, -0.54 pp [-0.96, -0.17] [DEC tests.N3.rows.1.diff,ci95].
- 200-record unit, flip(ref|gguf-bf16) minus flip(bf16-srv|fp8): 6 vs 51 flips, -1.88 pp [-2.42, -1.33] [DEC tests.N3.rows.2.diff,ci95].
- 200-record unit, flip(ref|gguf-bf16) minus flip(gguf-bf16|q8_0): 6 vs 21 flips, -0.62 pp [-1.04, -0.25] [DEC tests.N3.rows.3.diff,ci95].

**Status: not supported.** Against ref, an engine change at BF16 flips 0.24% (vLLM) and 0.21% (llama.cpp) of decisions; FP8 flips 1.20% and Q8_0 0.59%. On the 200-record unit every engine-minus-8-bit difference is negative with a CI below 0.

Rule: supported if each engine change at BF16 (transformers -> vLLM, transformers -> llama.cpp) flips at least as many decisions as each 8-bit quantization inside one engine (this is N3).

### H8

> H8 at equal memory, 4-bit Clef beats 8-bit Clef-flash on hard tasks but not easy ones.

- File sizes: Clef (27B) NVFP4 20.00 [SPD configs.clef.nvfp4.meta.file_size_gb] GB; Clef (27B) Q4_K_M 16.59 [SPD configs.clef.q4_k_m.meta.file_size_gb] GB; Clef-flash FP8 12.15 [SPD configs.clef-flash.fp8.meta.file_size_gb] GB; Clef-flash Q8_0 9.66 [SPD configs.clef-flash.q8_0.meta.file_size_gb] GB. Size ratios, larger over smaller as the analyzer defines them: NVFP4 / FP8 1.65; NVFP4 / Q8_0 2.07; Q4_K_M / FP8 1.36; Q4_K_M / Q8_0 1.72. Window: 15% [PUBALL equal_memory.tolerance] (two files are equal-memory when the larger is at most 1.15 times the smaller).
- Check: none of the 4 pairs of a 27B 4-bit build with a Clef-flash 8-bit build is within the window (closest: Q4_K_M / FP8 at 1.36), and PUBALL `equal_memory.pairs` (8 pairs) contains no such pair [PUBALL equal_memory.pairs].
- Supplementary, not pre-registered, outside the size window: Clef (27B) NVFP4 vs Clef-flash FP8, size ratio 1.65, accuracy of the 27B minus Clef-flash on the same questions: pooled hard tasks +5.22 pp [+4.30, +6.18] [HYP cross_size_4bit_vs_8bit.pairs.clef/nvfp4|clef-flash/fp8.pooled_hard.delta_accuracy_big_minus_small.delta,ci95], pooled easy tasks -0.97 pp [-1.34, -0.59] [HYP cross_size_4bit_vs_8bit.pairs.clef/nvfp4|clef-flash/fp8.pooled_easy.delta_accuracy_big_minus_small.delta,ci95], hard minus easy +6.19 pp [+5.19, +7.17] [HYP cross_size_4bit_vs_8bit.pairs.clef/nvfp4|clef-flash/fp8.hard_minus_easy.delta,ci95].
- Supplementary, not pre-registered, outside the size window: Clef (27B) NVFP4 vs Clef-flash Q8_0, size ratio 2.07, accuracy of the 27B minus Clef-flash on the same questions: pooled hard tasks +5.01 pp [+4.10, +5.95] [HYP cross_size_4bit_vs_8bit.pairs.clef/nvfp4|clef-flash/q8_0.pooled_hard.delta_accuracy_big_minus_small.delta,ci95], pooled easy tasks -1.17 pp [-1.54, -0.79] [HYP cross_size_4bit_vs_8bit.pairs.clef/nvfp4|clef-flash/q8_0.pooled_easy.delta_accuracy_big_minus_small.delta,ci95], hard minus easy +6.18 pp [+5.20, +7.19] [HYP cross_size_4bit_vs_8bit.pairs.clef/nvfp4|clef-flash/q8_0.hard_minus_easy.delta,ci95].
- Supplementary, not pre-registered, outside the size window: Clef (27B) Q4_K_M vs Clef-flash FP8, size ratio 1.36, accuracy of the 27B minus Clef-flash on the same questions: pooled hard tasks +6.44 pp [+5.54, +7.36] [HYP cross_size_4bit_vs_8bit.pairs.clef/q4_k_m|clef-flash/fp8.pooled_hard.delta_accuracy_big_minus_small.delta,ci95], pooled easy tasks -0.62 pp [-0.98, -0.25] [HYP cross_size_4bit_vs_8bit.pairs.clef/q4_k_m|clef-flash/fp8.pooled_easy.delta_accuracy_big_minus_small.delta,ci95], hard minus easy +7.06 pp [+6.10, +8.05] [HYP cross_size_4bit_vs_8bit.pairs.clef/q4_k_m|clef-flash/fp8.hard_minus_easy.delta,ci95].
- Supplementary, not pre-registered, outside the size window: Clef (27B) Q4_K_M vs Clef-flash Q8_0, size ratio 1.72, accuracy of the 27B minus Clef-flash on the same questions: pooled hard tasks +6.23 pp [+5.33, +7.16] [HYP cross_size_4bit_vs_8bit.pairs.clef/q4_k_m|clef-flash/q8_0.pooled_hard.delta_accuracy_big_minus_small.delta,ci95], pooled easy tasks -0.82 pp [-1.18, -0.47] [HYP cross_size_4bit_vs_8bit.pairs.clef/q4_k_m|clef-flash/q8_0.pooled_easy.delta_accuracy_big_minus_small.delta,ci95], hard minus easy +7.05 pp [+6.08, +8.03] [HYP cross_size_4bit_vs_8bit.pairs.clef/q4_k_m|clef-flash/q8_0.hard_minus_easy.delta,ci95].
- The hard/easy split behind those bullets (clef-flash ref accuracy, median split): hard = anli_r3, sst5, anli_r2, anli_r1, ragtruth, contractnli; easy = when2call, boolq, mmlu, banking77, clinc150, arc_challenge [HYP cross_size_4bit_vs_8bit.difficulty]. The split was chosen after the 27B results were seen.
- Sensitivity of the split: the primary median split above stays the reported one, and 3 alternate splits were added as a robustness check, also after the 27B results were seen: threshold_80 (hard = clef-flash ref accuracy below 80%, easy = the rest: hard = anli_r3, sst5, anli_r2, anli_r1, ragtruth, contractnli, when2call; easy = boolq, mmlu, banking77, clinc150, arc_challenge); median_mean_refs (median split on the mean of the clef-flash ref and clef (27B) ref accuracies: hard = anli_r3, sst5, anli_r2, anli_r1, ragtruth, when2call; easy = contractnli, mmlu, boolq, banking77, clinc150, arc_challenge); median_ex_boundary (primary median split with the two tasks nearest the median boundary swapped (hardest easy task to hard, easiest hard task to easy): hard = anli_r3, sst5, anli_r2, anli_r1, ragtruth, when2call; easy = contractnli, boolq, mmlu, banking77, clinc150, arc_challenge; the same hard and easy sets as median_mean_refs) [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity]. Range of the pooled deltas across the alternate splits, Clef (27B) minus Clef-flash on the same questions, with the split that gives each end: NVFP4 vs FP8: pooled hard +4.06 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/nvfp4|clef-flash/fp8.pooled_hard.delta_accuracy_big_minus_small.delta] (threshold_80) to +4.11 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/nvfp4|clef-flash/fp8.pooled_hard.delta_accuracy_big_minus_small.delta] (median_mean_refs), pooled easy -1.35 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/nvfp4|clef-flash/fp8.pooled_easy.delta_accuracy_big_minus_small.delta] (threshold_80) to -0.83 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/nvfp4|clef-flash/fp8.pooled_easy.delta_accuracy_big_minus_small.delta] (median_mean_refs); NVFP4 vs Q8_0: pooled hard +3.83 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/nvfp4|clef-flash/q8_0.pooled_hard.delta_accuracy_big_minus_small.delta] (threshold_80) to +3.89 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/nvfp4|clef-flash/q8_0.pooled_hard.delta_accuracy_big_minus_small.delta] (median_mean_refs), pooled easy -1.53 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/nvfp4|clef-flash/q8_0.pooled_easy.delta_accuracy_big_minus_small.delta] (threshold_80) to -1.01 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/nvfp4|clef-flash/q8_0.pooled_easy.delta_accuracy_big_minus_small.delta] (median_mean_refs); Q4_K_M vs FP8: pooled hard +4.94 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/q4_k_m|clef-flash/fp8.pooled_hard.delta_accuracy_big_minus_small.delta] (threshold_80) to +5.08 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/q4_k_m|clef-flash/fp8.pooled_hard.delta_accuracy_big_minus_small.delta] (median_mean_refs), pooled easy -0.92 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/q4_k_m|clef-flash/fp8.pooled_easy.delta_accuracy_big_minus_small.delta] (threshold_80) to -0.41 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/q4_k_m|clef-flash/fp8.pooled_easy.delta_accuracy_big_minus_small.delta] (median_mean_refs); Q4_K_M vs Q8_0: pooled hard +4.71 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/q4_k_m|clef-flash/q8_0.pooled_hard.delta_accuracy_big_minus_small.delta] (threshold_80) to +4.85 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/q4_k_m|clef-flash/q8_0.pooled_hard.delta_accuracy_big_minus_small.delta] (median_mean_refs), pooled easy -1.10 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.threshold_80.pairs.clef/q4_k_m|clef-flash/q8_0.pooled_easy.delta_accuracy_big_minus_small.delta] (threshold_80) to -0.59 pp [HYP cross_size_4bit_vs_8bit.difficulty_sensitivity.median_mean_refs.pairs.clef/q4_k_m|clef-flash/q8_0.pooled_easy.delta_accuracy_big_minus_small.delta] (median_mean_refs). The pattern (pooled hard delta above 0 and pooled easy delta at or below the pooled hard delta, on the point estimates) holds in all 12 pair and split combinations. These are robustness checks of a post-hoc split and are not pre-registered.
- Pairs that are inside the window: PUBALL `equal_memory.pairs` lists 6 pairs of a 27B 4-bit build with a Clef-flash BF16 build (2 further pairs involve ref-fla, a 2,400-question subset, and are skipped). These pair 27B 4-bit with Clef-flash BF16, which H8 does not ask about; they are listed for completeness and are not a test of H8.
- Inside the window: Clef (27B) Q4_K_M (16.59 GB) vs Clef-flash gguf-bf16 (18.16 GB), size ratio 1.095 [PUBALL equal_memory.pairs.0.size_ratio]: Clef-flash minus 27B accuracy -1.46 pp [-1.86, -1.09] [PUBALL equal_memory.pairs.0.delta_accuracy_b_minus_a.delta,ci95] on 31,275 questions.
- Inside the window: Clef (27B) Q4_K_M (16.59 GB) vs Clef-flash bf16-srv (19.06 GB), size ratio 1.149 [PUBALL equal_memory.pairs.1.size_ratio]: Clef-flash minus 27B accuracy -1.46 pp [-1.86, -1.10] [PUBALL equal_memory.pairs.1.delta_accuracy_b_minus_a.delta,ci95] on 31,275 questions.
- Inside the window: Clef (27B) Q4_K_M (16.59 GB) vs Clef-flash ref (19.06 GB), size ratio 1.149 [PUBALL equal_memory.pairs.2.size_ratio]: Clef-flash minus 27B accuracy -1.45 pp [-1.86, -1.08] [PUBALL equal_memory.pairs.2.delta_accuracy_b_minus_a.delta,ci95] on 31,275 questions.
- Inside the window: Clef (27B) NVFP4 (20.00 GB) vs Clef-flash gguf-bf16 (18.16 GB), size ratio 1.101 [PUBALL equal_memory.pairs.4.size_ratio]: Clef-flash minus 27B accuracy -0.83 pp [-1.23, -0.44] [PUBALL equal_memory.pairs.4.delta_accuracy_b_minus_a.delta,ci95] on 31,275 questions.
- Inside the window: Clef (27B) NVFP4 (20.00 GB) vs Clef-flash bf16-srv (19.06 GB), size ratio 1.049 [PUBALL equal_memory.pairs.5.size_ratio]: Clef-flash minus 27B accuracy -0.84 pp [-1.24, -0.44] [PUBALL equal_memory.pairs.5.delta_accuracy_b_minus_a.delta,ci95] on 31,275 questions.
- Inside the window: Clef (27B) NVFP4 (20.00 GB) vs Clef-flash ref (19.06 GB), size ratio 1.049 [PUBALL equal_memory.pairs.6.size_ratio]: Clef-flash minus 27B accuracy -0.82 pp [-1.23, -0.43] [PUBALL equal_memory.pairs.6.delta_accuracy_b_minus_a.delta,ci95] on 31,275 questions.

**Status: not evaluable.** No 27B 4-bit / Clef-flash 8-bit pair is within the 15% window. The closest is Clef (27B) Q4_K_M (16.59 GB) against Clef-flash FP8 (12.15 GB), a size ratio of 1.36 against a limit of 1.15. The comparison H8 pre-registers cannot be formed from the builds this sweep made, and no further run of this sweep changes that.

Rule: needs 27B 4-bit accuracy on the same records and a 4-bit Clef / 8-bit Clef-flash pair within the analyzer's 15% size window. No 4-bit Clef build lies within that window of an 8-bit Clef-flash build, so the pre-registered comparison cannot be formed with the builds this sweep made; the supplementary comparison is outside the window and is not pre-registered.

### H9

> H9 ordinal score questions shift toward the middle under 4-bit.

- sst5 (5-point scale, public), nvfp4: center pull -0.02673 [-0.03099, -0.02262] [HYP score_shift.per_config.nvfp4.center_pull.delta,ci95] (negative = toward the middle); mean signed shift +0.01921 [+0.01454, +0.02399] [HYP score_shift.per_config.nvfp4.mean_signed_shift.delta,ci95].
- sst5 (5-point scale, public), q4_k_m: center pull -0.01253 [-0.01659, -0.00861] [HYP score_shift.per_config.q4_k_m.center_pull.delta,ci95] (negative = toward the middle); mean signed shift +0.00272 [-0.00190, +0.00712] [HYP score_shift.per_config.q4_k_m.mean_signed_shift.delta,ci95].
- sst5 (5-point scale, public), fp8: center pull -0.00192 [-0.00322, -0.00053] [HYP score_shift.per_config.fp8.center_pull.delta,ci95] (negative = toward the middle); mean signed shift -0.00121 [-0.00253, +0.00020] [HYP score_shift.per_config.fp8.mean_signed_shift.delta,ci95].
- sst5 (5-point scale, public), q8_0: center pull +0.00074 [+0.00019, +0.00128] [HYP score_shift.per_config.q8_0.center_pull.delta,ci95] (negative = toward the middle); mean signed shift -0.00046 [-0.00101, +0.00006] [HYP score_shift.per_config.q8_0.mean_signed_shift.delta,ci95].
- Private r1 score6, nvfp4: center pull -0.0125 [-0.0198, -0.0053] [PRIV label_free.r1.score6.pairs.ref|nvfp4.score_shift.center_pull,center_pull_ci95]; mean signed shift +0.0206.
- Private r4a score5, nvfp4: center pull -0.0273 [-0.0304, -0.0241] [PRIV label_free.r4a.score5.pairs.ref|nvfp4.score_shift.center_pull,center_pull_ci95]; mean signed shift -0.0282.
- Private r1 score6, q4_k_m: center pull -0.0303 [-0.0371, -0.0232] [PRIV label_free.r1.score6.pairs.ref|q4_k_m.score_shift.center_pull,center_pull_ci95]; mean signed shift +0.0317.
- Private r4a score5, q4_k_m: center pull -0.1290 [-0.1323, -0.1259] [PRIV label_free.r4a.score5.pairs.ref|q4_k_m.score_shift.center_pull,center_pull_ci95]; mean signed shift -0.1398.

**Status: supported.** All 6 cells (two 4-bit builds x sst5, R1 score, R4a score5) show a pull toward the middle with CIs below 0; the effect is small, 0.013 to 0.129 of a scale point.

Rule: supported if the center-pull 95% CI (|expected score - middle|, quantized minus ref) lies below 0 for both 4-bit builds on sst5 and on every private score question (R1 6-point score, R4a score5 bucket).

### H10

> H10 a confidence-threshold cascade (4-bit first, escalate low-confidence cases to BF16) recovers near-BF16 agreement while escalating a small fraction.

- Public, nvfp4, escalate when its top probability < 0.65: escalation 13.9% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.3.escalation_rate], agreement with ref 99.12% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.3.agreement_with_ref] (standalone 95.44% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.agreement]), accuracy 82.00% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.3.accuracy] (ref 82.22% [PUB models.clef-flash.configs.ref.pooled.accuracy]).
- Public, nvfp4, escalate when its top probability < 0.7: escalation 17.7% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.4.escalation_rate], agreement with ref 99.45% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.4.agreement_with_ref] (standalone 95.44% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.agreement]), accuracy 82.06% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.4.accuracy] (ref 82.22% [PUB models.clef-flash.configs.ref.pooled.accuracy]).
- Public, nvfp4, escalate when its top probability < 0.8: escalation 27.4% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.6.escalation_rate], agreement with ref 99.86% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.6.agreement_with_ref] (standalone 95.44% [PUB models.clef-flash.configs.nvfp4.pooled.vs_ref.agreement]), accuracy 82.18% [PUB models.clef-flash.configs.nvfp4.cascade.pooled.6.accuracy] (ref 82.22% [PUB models.clef-flash.configs.ref.pooled.accuracy]).
- Public, q4_k_m, escalate when its top probability < 0.65: escalation 13.2% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.3.escalation_rate], agreement with ref 99.44% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.3.agreement_with_ref] (standalone 96.46% [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.agreement]), accuracy 82.16% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.3.accuracy] (ref 82.22% [PUB models.clef-flash.configs.ref.pooled.accuracy]).
- Public, q4_k_m, escalate when its top probability < 0.7: escalation 17.0% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.4.escalation_rate], agreement with ref 99.66% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.4.agreement_with_ref] (standalone 96.46% [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.agreement]), accuracy 82.18% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.4.accuracy] (ref 82.22% [PUB models.clef-flash.configs.ref.pooled.accuracy]).
- Public, q4_k_m, escalate when its top probability < 0.8: escalation 26.5% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.6.escalation_rate], agreement with ref 99.90% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.6.agreement_with_ref] (standalone 96.46% [PUB models.clef-flash.configs.q4_k_m.pooled.vs_ref.agreement]), accuracy 82.19% [PUB models.clef-flash.configs.q4_k_m.cascade.pooled.6.accuracy] (ref 82.22% [PUB models.clef-flash.configs.ref.pooled.accuracy]).
- Private R4a+R4b (the gold-bearing items the analyzer pairs), nvfp4, t = 0.5: escalation 26.2% [PRIVA models.clef-flash.configs.nvfp4.cascade.pooled.0.escalation_rate], agreement with ref 99.17% [PRIVA models.clef-flash.configs.nvfp4.cascade.pooled.0.agreement_with_ref].
- Private R4a+R4b (the gold-bearing items the analyzer pairs), nvfp4, t = 0.6: escalation 40.0% [PRIVA models.clef-flash.configs.nvfp4.cascade.pooled.2.escalation_rate], agreement with ref 99.92% [PRIVA models.clef-flash.configs.nvfp4.cascade.pooled.2.agreement_with_ref].
- Private R4a+R4b (the gold-bearing items the analyzer pairs), q4_k_m, t = 0.5: escalation 29.6% [PRIVA models.clef-flash.configs.q4_k_m.cascade.pooled.0.escalation_rate], agreement with ref 99.26% [PRIVA models.clef-flash.configs.q4_k_m.cascade.pooled.0.agreement_with_ref].
- Private R4a+R4b (the gold-bearing items the analyzer pairs), q4_k_m, t = 0.6: escalation 43.6% [PRIVA models.clef-flash.configs.q4_k_m.cascade.pooled.2.escalation_rate], agreement with ref 99.92% [PRIVA models.clef-flash.configs.q4_k_m.cascade.pooled.2.agreement_with_ref].

**Status: mixed.** Public: t = 0.7 escalates 17-18% of questions and lifts agreement to 99.5-99.7% (the 8-bit builds agree 98.8% and 99.4% standalone). Private R4a/R4b: the same 99% needs t = 0.5 and escalates 26-30% (t = 0.6: 40-44%); there are many more low-confidence questions.

Rule: this report's operational reading: near-BF16 agreement = at least 99% top-1 agreement with ref (the level of the two 8-bit builds), small fraction = at most 20% escalated, checked at t = 0.7 (public) and t = 0.5 (private R4a/R4b, where it is the lowest threshold tabulated). supported if both public and private reach that; mixed if only one does.

### H11

> H11 GGUF Q4_K_M and NVFP4 differ at the same nominal bit width.

- Q4_K_M vs NVFP4 directly, full public set: they disagree on 1,498 [HYP q4_pair.flips_between] questions (4.79% [4.55, 5.02] [HYP q4_pair.flip_rate_between,flip_rate_between_ci95]); each differs from ref on 3.54% [HYP q4_pair.flip_rate_vs_ref.q4_k_m] (Q4_K_M) and 4.56% [HYP q4_pair.flip_rate_vs_ref.nvfp4] (NVFP4) of questions.
- Accuracy, NVFP4 minus Q4_K_M: -0.59 pp [-0.81, -0.37] [HYP q4_pair.delta_accuracy_nvfp4_minus_q4_k_m.delta,ci95]; Brier +0.00623 [+0.00448, +0.00797] [HYP q4_pair.delta_brier_nvfp4_minus_q4_k_m.delta,ci95]; ECE +0.00229 [-0.00011, +0.00469] [HYP q4_pair.delta_ece_nvfp4_minus_q4_k_m.delta,ci95].
- Q4_K_M is the more accurate build on 10 of 12 tasks, NVFP4 on 1 [HYP q4_pair.per_task].
- Same nominal bit width is not the same size: NVFP4 9.13 [SPD configs.clef-flash.nvfp4.meta.file_size_gb] GB vs Q4_K_M 5.72 [SPD configs.clef-flash.q4_k_m.meta.file_size_gb] GB.

**Status: supported.** They differ measurably: Q4_K_M is the more accurate and the better scored (Brier) of the two on this benchmark, and it changes fewer decisions against ref than NVFP4 does.

Rule: supported if the paired 95% CI of the accuracy difference between the two 4-bit builds excludes 0.

## Report tests N1-N5

### N1

> N1 FP8's flip rate against `bf16-srv` is not significantly larger than `bf16-srv` against `bf16-srv-r2` (8-bit inside restart noise; H1 restated against a measured floor).

- flip(bf16-srv, fp8) = 51 of 2,400 questions (2.12% [1.54, 2.71] [DEC pairs.bf16-srv|fp8.flip_rate,flip_ci95]); flip(bf16-srv, bf16-srv-r2) = 0 (0.00% [0.00, 0.00] [DEC pairs.bf16-srv|bf16-srv-r2.flip_rate,flip_ci95]; exact 95% 0.00-0.15%).
- Difference +2.12 pp [+1.54, +2.71] [DEC tests.N1.diff,ci95]; stratified-by-task bootstrap [+1.58, +2.71] [DEC tests.N1.ci95_stratified].
- Supplementary, not pre-registered: against FP8's own restart (5 flips) the difference is +1.92 pp [+1.33, +2.50] [DEC tests.N1.supplementary_not_preregistered.diff,ci95].

**Status: not supported.** FP8 flips 51 decisions where the BF16 restart flips 0; the interval of the difference excludes 0, and FP8's own restart noise (5 flips) is 10x smaller than its effect.

Rule: supported if FP8's flip rate against bf16-srv is NOT significantly larger than the bf16-srv restart flip rate, i.e. the 95% CI of the difference does not lie above 0

### N2

> N2 NVFP4 and Q4_K_M flip rates exceed every non-quantization source, with intervals that exclude them.

- 16 comparisons (NVFP4 vs bf16-srv and Q4_K_M vs gguf-bf16, each minus the 8 non-quantization flip rates: four restart pairs, ref-r2, ref-fla, both engine changes). Flip counts on 2,400 questions: NVFP4 139, Q4_K_M 139 [DEC pairs.bf16-srv|nvfp4.flips]; the largest non-quantization source is the NVFP4 restart with 20 flips [DEC pairs.nvfp4|nvfp4-r2.flips].
- Every CI excludes 0 (lower bound > 0): True [DEC tests.N2.all_lower_gt_zero]; with stratified resampling: True; with Bonferroni (k = 16): True [DEC tests.N2.all_lower_gt_zero_bonferroni].
- Weakest row: bf16-srv|nvfp4 minus nvfp4|nvfp4-r2: +4.96 pp [+4.00, +5.96] [DEC tests.N2.weakest_row.diff,ci95] (Bonferroni lower bound +3.58 pp).

**Status: supported.** Both 4-bit builds change 5.8% of decisions on this unit against at most 0.8% for any non-quantization source.

Rule: supported if every quantization flip rate (NVFP4 vs bf16-srv, Q4_K_M vs gguf-bf16) exceeds every non-quantization source with a 95% CI of the difference that excludes 0 (lower bound > 0)

### N3

> N3 Changing engine at BF16 (`ref` -> `bf16-srv`, `ref` -> `gguf-bf16`) moves at least as many decisions as 8-bit quantization within one engine (H7).

- flip(ref|bf16-srv) - flip(bf16-srv|fp8): 8 vs 51 flips, -1.79 pp [-2.33, -1.25] [DEC tests.N3.rows.0.diff,ci95]; CI entirely below 0: True.
- flip(ref|bf16-srv) - flip(gguf-bf16|q8_0): 8 vs 21 flips, -0.54 pp [-0.96, -0.17] [DEC tests.N3.rows.1.diff,ci95]; CI entirely below 0: True.
- flip(ref|gguf-bf16) - flip(bf16-srv|fp8): 6 vs 51 flips, -1.88 pp [-2.42, -1.33] [DEC tests.N3.rows.2.diff,ci95]; CI entirely below 0: True.
- flip(ref|gguf-bf16) - flip(gguf-bf16|q8_0): 6 vs 21 flips, -0.62 pp [-1.04, -0.25] [DEC tests.N3.rows.3.diff,ci95]; CI entirely below 0: True.

**Status: not supported.** All four engine-minus-8-bit differences are negative and significant: on this unit an engine change at BF16 moves 0.25%-0.33% of decisions, FP8 2.12% and Q8_0 0.88%. (Same as H7.)

Rule: supported if each engine change at BF16 flips at least as many decisions as each 8-bit quantization inside one engine: point estimate of (engine - 8-bit) >= 0; contradicted for a combination when the 95% CI of the difference lies below 0

### N4

> N4 Kernel selection (`ref` vs `ref-fla`) flips a nonzero number of decisions, within the range of the vLLM restart drift.

- ref vs ref-fla: 10 [DEC tests.N4.ref_fla_flips] flips of 2,400 (0.42% [DEC tests.N4.ref_fla_flip_rate]); nonzero: True.
- vLLM restart flip counts: BF16 0, FP8 5, NVFP4 20; range 0 to 20 [DEC tests.N4.vllm_restart_range]. Within that range: True. Against the BF16 restart alone (0 flips) it is above: True.
- Differences of flip rates, ref-fla minus each vLLM restart: bf16-srv|bf16-srv-r2: +0.42 pp [+0.21, +0.71] [DEC tests.N4.differences_vs_each_vllm_restart.0.diff,ci95]; fp8|fp8-r2: +0.21 pp [-0.08, +0.54] [DEC tests.N4.differences_vs_each_vllm_restart.1.diff,ci95]; nvfp4|nvfp4-r2: -0.42 pp [-0.87, +0.00] [DEC tests.N4.differences_vs_each_vllm_restart.2.diff,ci95].

**Status: supported.** The kernel switch alone flips 10 of 2,400 decisions: more than the FP8 restart (5), fewer than the NVFP4 restart (20), and significantly above the BF16 vLLM restart (0). Whether it is 'within the range' depends on which restart pairs make up 'the vLLM restart drift'.

Rule: supported if ref-fla flips a nonzero number of decisions against ref AND that count lies within the range of the vLLM restart flip counts (the three measured vLLM restart pairs)

### N5

> N5 llama.cpp at BF16 is bit-identical across restarts (`gguf-bf16` vs `gguf-bf16-r2`).

- gguf-bf16 vs gguf-bf16-r2: 2,400 [DEC tests.N5.n_identical_records] of 2,400 [DEC tests.N5.n_records] records bit-identical (100.0% [DEC tests.N5.share_identical]); basis: probs (llama.cpp returns full-precision probabilities and no logits); flips 0, max |dprob| 0, mean TVD 0 [DEC tests.N5.max_dprob].
- For contrast, the vLLM restarts are bit-identical on 93.7% [DEC pairs.bf16-srv|bf16-srv-r2.share_identical] (BF16), 15.0% [DEC pairs.fp8|fp8-r2.share_identical] (FP8) and 76.8% [DEC pairs.nvfp4|nvfp4-r2.share_identical] (NVFP4) of records.

**Status: supported.** All 2,400 records are identical, so the llama.cpp restart noise floor is exactly zero on this unit.

Rule: supported if every record of gguf-bf16 and gguf-bf16-r2 is bit-identical

## 27B

Clef (27B) public-set results [PUBALL models.clef.configs]: complete on all 12 public tasks for ref, bf16-srv, fp8, gguf-bf16, nvfp4, q4_k_m, q8_0; ref-fla on 2,400 questions only. `ref` is the BF16 reference (transformers-native, transformers 5.16.1, batch 1) [SPD configs.clef.ref.meta]; every other 27B config is compared with the 27B's own `ref`. The table below sets the 27B `ref` against the 9B `ref` in absolute terms. Tasks finished: anli_r1, anli_r2, anli_r3, arc_challenge, banking77, boolq, clinc150, contractnli, mmlu, ragtruth, sst5, when2call; none missing [HYP big_vs_small.tasks_missing]. Pairing is by record id; intervals are paired bootstraps over questions.

| task | n | 27B accuracy | 9B accuracy | 27B minus 9B pp [95% CI] | top-1 agreement 27B vs 9B | macro-F1 27B / 9B |
|---|---|---|---|---|---|---|
| anli_r1 | 1,000 | 70.80% | 67.30% | +3.50 [+0.60, +6.50] | 73.2% | 0.705 / 0.671 |
| anli_r2 | 1,000 | 64.10% | 58.30% | +5.80 [+2.80, +9.00] | 70.4% | 0.640 / 0.581 |
| anli_r3 | 1,200 | 56.58% | 51.58% | +5.00 [+2.25, +8.08] | 67.6% | 0.565 / 0.512 |
| arc_challenge | 1,172 | 97.95% | 98.55% | -0.60 [-1.45, +0.26] | 97.7% | 0.980 / 0.986 |
| banking77 | 3,080 | 94.25% | 95.39% | -1.14 [-1.79, -0.52] | 96.4% | 0.942 / 0.954 |
| boolq | 3,270 | 91.19% | 91.13% | +0.06 [-0.80, +0.92] | 93.6% | 0.905 / 0.906 |
| clinc150 | 5,000 | 97.30% | 98.00% | -0.70 [-1.10, -0.30] | 97.5% | 0.976 / 0.985 |
| contractnli | 1,991 | 77.35% | 74.08% | +3.26 [+1.61, +4.82] | 84.7% | 0.745 / 0.721 |
| mmlu | 5,000 | 89.68% | 91.46% | -1.78 [-2.64, -0.94] | 88.4% | 0.896 / 0.914 |
| ragtruth | 2,700 | 81.11% | 67.37% | +13.74 [+11.96, +15.59] | 73.2% | 0.784 / 0.466 |
| sst5 | 2,210 | 57.69% | 58.05% | -0.36 [-2.17, +1.58] | 78.1% | 0.538 / 0.560 |
| when2call | 3,652 | 75.55% | 74.37% | +1.18 [-0.11, +2.46] | 82.5% | 0.571 / 0.554 |

Source for every row: [HYP big_vs_small.per_task.<task>]. Pooled over the 12 finished public tasks (31,275 questions): 27B 83.69% [HYP big_vs_small.pooled.acc_27b], 9B 82.22% [HYP big_vs_small.pooled.acc_9b], difference +1.47 pp [+1.09, +1.85] [HYP big_vs_small.pooled.delta_accuracy_27b_minus_9b.delta,ci95] (pooled over questions, not stratified by task; both figures cover the same tasks).

Reading (from the intervals above): the 27B is ahead on anli_r1, anli_r2, anli_r3, contractnli, ragtruth; level on arc_challenge, boolq, sst5, when2call; behind on banking77, clinc150, mmlu. Its top-1 agreement with the 9B reference ranges 68% to 98% by task; pooled it is 86.7% [HYP big_vs_small.pooled.top1_agreement_27b_vs_9b], below the lowest pooled agreement of a 27B build with its own ref (95.2%, nvfp4) [PUBALL models.clef.configs.nvfp4.pooled.vs_ref.agreement].

Speed (quality-run throughput, not the latency benchmark): 27B / 9B per-record time 3.14 [SPD ratio_27b_over_9b.pooled_with_sidecar_fallback.pooled_ratio] pooled over 16 sets (per-set range 3.01 to 3.32); the four 9B sets without an END line use the per-set sidecar [SPD ratio_27b_over_9b.fallback_sets].

- Private r4a / choice5: agreement with the hosted label, 27B ref 92.8% (macro-F1 0.266) vs Clef-flash ref 90.0% (macro-F1 0.235) [PRIV label_agreement.r4a.choice5.models].
- Private r4a / score5: agreement with the hosted label, 27B ref 66.6% (macro-F1 0.241) vs Clef-flash ref 56.9% (macro-F1 0.256) [PRIV label_agreement.r4a.score5.models].
- Private r4b / choice3: agreement with the hosted label, 27B ref 92.2% (macro-F1 0.782) vs Clef-flash ref 65.3% (macro-F1 0.455) [PRIV label_agreement.r4b.choice3.models].

### 27B quantization

Each 27B config against the 27B `ref`, full public set (31,275 questions), paired bootstrap over questions; the last column is the Clef-flash flip count for the same config against the Clef-flash `ref`. `ref-fla` is left out because the 27B has it on a subset only.

| config | file size GB | accuracy change vs 27B ref pp [95% CI] | flips (% of questions) | top-1 agreement | Clef-flash flips (% of questions) |
|---|---|---|---|---|---|
| bf16-srv | 54.97 | -0.01 [-0.08, +0.06] | 155 (0.50%) | 99.50% | 75 (0.24%) |
| gguf-bf16 | 54.06 | -0.02 [-0.10, +0.05] | 156 (0.50%) | 99.50% | 67 (0.21%) |
| fp8 | 30.65 | -0.11 [-0.23, +0.02] | 520 (1.66%) | 98.34% | 374 (1.20%) |
| q8_0 | 28.73 | -0.07 [-0.17, +0.02] | 264 (0.84%) | 99.16% | 183 (0.59%) |
| nvfp4 | 20.00 | -0.65 [-0.87, -0.42] | 1,502 (4.80%) | 95.20% | 1,426 (4.56%) |
| q4_k_m | 16.59 | -0.02 [-0.20, +0.16] | 1,000 (3.20%) | 96.80% | 1,108 (3.54%) |

Source for every row: [SPD configs.clef.<cfg>.meta.file_size_gb]; [PUBALL models.clef.configs.<cfg>.pooled.vs_ref.delta_accuracy.delta,ci95], [PUBALL models.clef.configs.<cfg>.pooled.vs_ref.n_flips] and [PUBALL models.clef.configs.<cfg>.pooled.vs_ref.agreement]; Clef-flash flips [PUBALL models.clef-flash.configs.<cfg>.pooled.vs_ref.n_flips]. Rerun floor on the same questions, ref against ref-r2: Clef (27B) 130 [PUBALL models.clef.noise_floor.ref.ref-r2.pooled.n_flips] flips, Clef-flash 66 [PUBALL models.clef-flash.noise_floor.ref.ref-r2.pooled.n_flips] flips.

Private sets R1, R2, R4a and R4b pooled (27,738 questions for the 27B, 27,738 for Clef-flash), flips against each model's own `ref`:

| config | Clef (27B) flips (% of questions) | Clef-flash flips (% of questions) |
|---|---|---|
| bf16-srv | 157 (0.57%) | 121 (0.44%) |
| gguf-bf16 | 122 (0.44%) | 88 (0.32%) |
| fp8 | 467 (1.68%) | 606 (2.18%) |
| q8_0 | 244 (0.88%) | 249 (0.90%) |
| nvfp4 | 1,195 (4.31%) | 2,168 (7.82%) |
| q4_k_m | 917 (3.31%) | 2,425 (8.74%) |
| ref-r2 (rerun floor) | 106 (0.38%) | 102 (0.37%) |

Source for every row: sums over the four sets of [PRIV label_free_big.<set>.pooled.pairs.ref|<cfg>.flips] (27B) and [PRIV label_free.<set>.pooled.pairs.ref|<cfg>.flips] (Clef-flash); denominators from [PRIV label_free_big.<set>.pooled.n_questions] and [PRIV label_free.<set>.pooled.n_questions].

- Private r4a / choice5: change in agreement with the hosted label, 27B config against the 27B ref (paired): fp8: -0.11 pp [-0.20, -0.01] [PRIV label_agreement.r4a.choice5.models.clef.configs.fp8.delta_vs_ref.delta_accuracy.delta,ci95]; q8_0: -0.03 pp [-0.11, +0.05] [PRIV label_agreement.r4a.choice5.models.clef.configs.q8_0.delta_vs_ref.delta_accuracy.delta,ci95]; nvfp4: -0.11 pp [-0.29, +0.08] [PRIV label_agreement.r4a.choice5.models.clef.configs.nvfp4.delta_vs_ref.delta_accuracy.delta,ci95]; q4_k_m: -0.19 pp [-0.32, -0.05] [PRIV label_agreement.r4a.choice5.models.clef.configs.q4_k_m.delta_vs_ref.delta_accuracy.delta,ci95].
- Private r4a / score5: change in agreement with the hosted label, 27B config against the 27B ref (paired): fp8: -0.51 pp [-0.84, -0.16] [PRIV label_agreement.r4a.score5.models.clef.configs.fp8.delta_vs_ref.delta_accuracy.delta,ci95]; q8_0: -0.15 pp [-0.40, +0.13] [PRIV label_agreement.r4a.score5.models.clef.configs.q8_0.delta_vs_ref.delta_accuracy.delta,ci95]; nvfp4: -0.56 pp [-1.09, -0.04] [PRIV label_agreement.r4a.score5.models.clef.configs.nvfp4.delta_vs_ref.delta_accuracy.delta,ci95]; q4_k_m: -0.90 pp [-1.39, -0.39] [PRIV label_agreement.r4a.score5.models.clef.configs.q4_k_m.delta_vs_ref.delta_accuracy.delta,ci95].
- Private r4b / choice3: change in agreement with the hosted label, 27B config against the 27B ref (paired): fp8: -0.60 pp [-1.80, +0.00] [PRIV label_agreement.r4b.choice3.models.clef.configs.fp8.delta_vs_ref.delta_accuracy.delta,ci95]; q8_0: +0.00 pp [-1.80, +1.80] [PRIV label_agreement.r4b.choice3.models.clef.configs.q8_0.delta_vs_ref.delta_accuracy.delta,ci95]; nvfp4: -1.80 pp [-4.79, +1.20] [PRIV label_agreement.r4b.choice3.models.clef.configs.nvfp4.delta_vs_ref.delta_accuracy.delta,ci95]; q4_k_m: +1.20 pp [-1.20, +3.59] [PRIV label_agreement.r4b.choice3.models.clef.configs.q4_k_m.delta_vs_ref.delta_accuracy.delta,ci95].

## Private sets (aggregates only)

Full tables: `analysis/private_aggregates.md` / `.json`; analyzer view: `analysis/flash_private/`. R1/R2 have no trustworthy gold, so only label-free numbers exist for them. The hosted model's recorded label is run a of the hosted repeat (verified: the recorded label equals run a on every repeat pair).

- R4a choice5: the hosted model disagrees with its own repeat on 7.71% [6.97, 8.44] [PRIV hosted_repeat.r4a.choice5.disagreement,disagreement_ci95] of 5,319 pairs (410 pairs); PLAN.md quotes 7.7%; reproduced: True [PRIV hosted_repeat.r4a.choice5.reproduces_plan_quote]. Macro-F1 of the repeat against the recorded label 0.348 [PRIV hosted_repeat.r4a.choice5.run_b_vs_run_a.macro_f1].
- R4a score5: the hosted model disagrees with its own repeat on 28.05% [26.87, 29.25] [PRIV hosted_repeat.r4a.score5.disagreement,disagreement_ci95] of 5,319 pairs (1492 pairs); PLAN.md quotes 28.1%; reproduced: True [PRIV hosted_repeat.r4a.score5.reproduces_plan_quote]. Macro-F1 of the repeat against the recorded label 0.484 [PRIV hosted_repeat.r4a.score5.run_b_vs_run_a.macro_f1].
- R4a choice5, on the 5,319 repeat records, agreement with the hosted label: majority-class baseline 94.0%, hosted repeat 92.3% (macro-F1 0.348), Clef-flash ref 90.7% (macro-F1 0.240), NVFP4 90.4%, Q4_K_M 89.7%, Q8_0 90.7% [PRIV label_agreement.r4a.choice5.hosted_repeat_subset]. Clef-flash ref minus hosted repeat: -1.58 pp [-2.33, -0.83] [PRIV label_agreement.r4a.choice5.hosted_repeat_subset.models.clef-flash.configs.ref.delta_vs_hosted_run_b.delta_accuracy.delta,ci95] agreement, -10.87 pp [-21.21, -7.47] [PRIV label_agreement.r4a.choice5.hosted_repeat_subset.models.clef-flash.configs.ref.delta_vs_hosted_run_b.delta_macro_f1.delta,ci95] macro-F1.
- R4a score5, on the 5,319 repeat records, agreement with the hosted label: majority-class baseline 69.9%, hosted repeat 71.9% (macro-F1 0.484), Clef-flash ref 57.1% (macro-F1 0.247), NVFP4 54.9%, Q4_K_M 49.6%, Q8_0 57.2% [PRIV label_agreement.r4a.score5.hosted_repeat_subset]. Clef-flash ref minus hosted repeat: -14.81 pp [-16.47, -13.12] [PRIV label_agreement.r4a.score5.hosted_repeat_subset.models.clef-flash.configs.ref.delta_vs_hosted_run_b.delta_accuracy.delta,ci95] agreement, -23.69 pp [-30.16, -16.74] [PRIV label_agreement.r4a.score5.hosted_repeat_subset.models.clef-flash.configs.ref.delta_vs_hosted_run_b.delta_macro_f1.delta,ci95] macro-F1.
- Reading, R4a choice5: the majority-class baseline (94.0%) beats even the hosted repeat (92.3%) on raw agreement, so agreement alone says nothing; per-class recall separates the predictors. Recall of the second, third and fourth class: hosted repeat 31%, 30%, 18%; Clef-flash ref 2%, 10%, 9%; 27B ref 1%, 11%, 21% (supports 180, 105, 34) [PRIV label_agreement.r4a.choice5.hosted_repeat_subset]. Clef-flash is not a replacement for the hosted call on R4a choice5 or score5, in BF16 or 8-bit; quantization moves these aggregates by far less than that gap.
- Quantization effect on R4a choice5 agreement with the hosted label (change against ref, paired): fp8: -0.19 pp [-0.36, +0.00] [PRIV label_agreement.r4a.choice5.models.clef-flash.configs.fp8.delta_vs_ref.delta_accuracy.delta,ci95]; nvfp4: -0.20 pp [-0.52, +0.13] [PRIV label_agreement.r4a.choice5.models.clef-flash.configs.nvfp4.delta_vs_ref.delta_accuracy.delta,ci95]; q8_0: -0.03 pp [-0.16, +0.11] [PRIV label_agreement.r4a.choice5.models.clef-flash.configs.q8_0.delta_vs_ref.delta_accuracy.delta,ci95]; q4_k_m: -1.01 pp [-1.35, -0.67] [PRIV label_agreement.r4a.choice5.models.clef-flash.configs.q4_k_m.delta_vs_ref.delta_accuracy.delta,ci95].
- R1 and R2, label-free, flip rate against ref pooled over all questions: R1 q8_0: 1.3% [0.9, 1.6] [PRIV label_free.r1.pooled.pairs.ref|q8_0.flip_rate,flip_ci95]; R1 nvfp4: 10.4% [9.5, 11.3] [PRIV label_free.r1.pooled.pairs.ref|nvfp4.flip_rate,flip_ci95]; R1 q4_k_m: 9.6% [8.7, 10.4] [PRIV label_free.r1.pooled.pairs.ref|q4_k_m.flip_rate,flip_ci95]; R2 q8_0: 0.8% [0.6, 1.0] [PRIV label_free.r2.pooled.pairs.ref|q8_0.flip_rate,flip_ci95]; R2 nvfp4: 7.2% [6.6, 7.7] [PRIV label_free.r2.pooled.pairs.ref|nvfp4.flip_rate,flip_ci95]; R2 q4_k_m: 6.5% [6.0, 7.1] [PRIV label_free.r2.pooled.pairs.ref|q4_k_m.flip_rate,flip_ci95]; rerun floor (ref-r2): R1 0.63% [0.41, 0.86] [PRIV label_free.r1.pooled.pairs.ref|ref-r2.flip_rate,flip_ci95]; R2 0.19% [0.10, 0.29] [PRIV label_free.r2.pooled.pairs.ref|ref-r2.flip_rate,flip_ci95].

## Data anomalies and caveats found while computing these numbers

1. **PLAN.md record counts.** PLAN.md says held-out splits of 'about 500 records each'; the public task files hold banking77 3,080, clinc150 5,000, mmlu 5,000, when2call 3,652 and so on, 31,275 questions in total [MAN tasks.<task>.written]. All numbers here use what is on disk.
2. **Untagged reference runs (provenance gap, not verified from the outputs).** `ref` result lines for anli_r1, anli_r2, anli_r3, arc_challenge, banking77, clinc150 carry engine_version 'transformers 5.16.1' without the `gdn=` kernel tag that the later runs record [DEC conditions.ref.tasks_by_value]. PLAN.md says the container had no flash-linear-attention then, so they ran the PyTorch fallback, and the config-level meta.json says gdn_kernel = torch [SPD configs.clef-flash.ref.meta.gdn_kernel], but that file was written afterwards from the config definition, not observed at run time. The outputs cannot prove it either way: the share of bit-identical records between ref and ref-fla is 0.0-12.0% on the untagged tasks and 1.5-17.5% on the tagged ones [DEC pairs.ref|ref-fla.share_identical_by_task] (no untagged task is anywhere near 100%, as it would be if ref had run the same kernel as ref-fla and that kernel were reproducible, which has not been tested). The N4 comparison and every ref-based number assume the fallback.
3. **Timing gaps.** The sweep log has no END line for clef-flash/ref/anli_r1, clef-flash/ref/anli_r2, clef-flash/ref/anli_r3, clef-flash/ref/arc_challenge [SPD missing_end_lines]; per-config totals exclude them (the like-for-like table uses common sets), and the 27B/9B ratio takes the 9B time for those sets from the per-set sidecar (`wall_seconds`, load time not included).
4. **Analyzer columns that are not throughput.** `p50 ms`, `p95 ms` and `dec/s` in `analysis/flash_public/tables.md` come from per-request latency; for vLLM those requests ran 16 in flight, so dec/s there (bf16-srv: 0.30 [PUB models.clef-flash.configs.bf16-srv.pooled.latency.decisions_per_sec]) is not a throughput. Use `analysis/speed.md`.
5. **Analyzer limits fixed or worked around.** (a) `--models clef-flash` keeps the 27B out of `flash_public` (it was unfinished when that directory was built; test added); `analysis/all_public/` is the same analyzer over both models and is the source of every 27B quantization number. (b) For private sets the analyzer pairs only items with gold, so R1/R2 get no paired numbers, and it blends the two R4a questions into one macro-F1: `bench/private_agg.py` computes per-question and label-free numbers instead. (c) The analyzer's `noise_floor_max_*` annotation takes the maximum over per-task blocks of every repeat of the model (mixing engines); the decomposition uses matched pairs instead. (d) `ref-fla` appears in the analyzer's equal-memory table with accuracy on its 2,400-question subset next to full-set rows; do not compare them.
6. **'Bit-identical' has two bases.** Native and vLLM logits are on bf16 granularity and are compared as logits; llama.cpp emits full-precision probabilities and no logits, so its records are compared on probabilities. Raw max |dlogit| contains any constant offset of the logit scale (large for NVFP4 vs BF16), so the centred value is reported next to it.
7. **Rare classes.** R4a choice5 has classes with 3 and 45 records and R4b's 3-way choice a class with 1; all-class macro-F1 intervals are approximate there, and a macro-F1 over classes with support of at least 30 is reported beside it.
8. **Zero-flip pairs.** The percentile bootstrap interval of a flip rate collapses to [0, 0] when a pair has 0 flips; exact Clopper-Pearson intervals are shown next to it (0 of 2,400 gives an upper bound of 0.15%).
9. **27B scope.** Clef (27B) has results for ref, bf16-srv, fp8, gguf-bf16, nvfp4, q4_k_m, q8_0 on all 12 public tasks [PUBALL models.clef.configs]; ref-fla only on 2,400 public questions (a subset: never compare it with a full-set row). File sizes are recorded for every 27B config [SPD file_sizes_missing]. Every 27B quantization number is measured against the 27B's own ref; the cross-size comparison H8 asks for cannot be formed from this sweep's builds (see H8).

## Reproduce

```
python -m bench.analyze --records data/records --results results --out analysis/flash_public --models clef-flash
python -m bench.analyze --records data/records --results results --out analysis/all_public
python -m bench.analyze --private private --out analysis/flash_private --models clef-flash
python -m bench.decompose --out analysis
python -m bench.speed
python -m bench.hypotheses
python -m bench.private_agg
python -m bench.findings --root . --out analysis/findings.md
```

