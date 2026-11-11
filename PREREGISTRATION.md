# Pre-registration

Extracted word for word from the project's plan file, which also holds operational notes that are not published. The hypotheses are dated 2026-10-15 and the report tests 2026-10-17. Omitted passages are marked. These dates come from the plan file itself. There is no earlier public timestamp.

## Pre-registered hypotheses (written before any results; published as-is with the outcome)

Obvious: H1 8-bit is indistinguishable from BF16 within the noise floor. H2 4-bit loses accuracy, more on
Clef-flash than Clef in absolute flips. H3 flips concentrate where ref confidence is low.
Less obvious: H4 calibration (ECE/Brier) degrades before accuracy does. H5 out-of-scope detection degrades
more than in-scope classification. H6 drift grows with input length, so real-world data (longer, messier)
shows more drift than benchmarks. H7 engine change (transformers -> vLLM / llama.cpp) moves decisions as
much as 8-bit quantization. H8 at equal memory, 4-bit Clef beats 8-bit Clef-flash on hard tasks but not
easy ones. H9 ordinal score questions shift toward the middle under 4-bit. H10 a confidence-threshold
cascade (4-bit first, escalate low-confidence cases to BF16) recovers near-BF16 agreement while escalating
a small fraction. H11 GGUF Q4_K_M and NVFP4 differ at the same nominal bit width.

## Write-up decision (owner, 2026-10-17; fixed before any quantization result was analysed)

Prior-art check 2026-10-17: the 8-bit-fine / 4-bit-costs-a-little headline already appears on several community
model cards, so the full cross-format comparison is released as data and an appendix, not as the thesis.

1. **arXiv technical report (the spine): the noise floor of a quantization comparison.** When a quantized model is
   compared with BF16, three non-quantization sources also move the decisions: the engine (transformers -> vLLM /
   llama.cpp), a restart of the same server with the same flags, and the kernel a library selects silently
   (transformers picks flash-linear-attention for the Gated DeltaNet layers whenever it is importable). The report
   measures each and states every quantization effect as the excess over them, on the same records. Public data
   only, so it is reproducible. Closest prior work, cited and positioned against: Dutta et al. 2024 "Accuracy is Not
   All You Need" (flips as the metric; arXiv:2407.09141), Masoudian et al. 2026 (backend explains ~39% of benchmark
   variability; no quantization, restarts or kernel selection; arXiv:2608.04714), batch-invariance work on
   nondeterminism within one running server. No paper found (broad search, 2026-10-17) that measures restart-to-
   restart variance or sets it against a quantization effect; the silent-kernel point was not searched specifically.

[Item 2, on a separate blog post, is omitted because it describes the private data.]

Comparison unit: the non-quantization sources exist on the first 200 records of each public task (`-r2`, `ref-fla`),
so the decomposition uses those same records for every config: top-1 flip rate against `ref`, mean TVD, max |dprob|,
paired bootstrap 95% intervals over records. Full-set numbers stay in the appendix.

Pre-registered for the report (in addition to H1-H11; outcomes published as-is):
- N1 FP8's flip rate against `bf16-srv` is not significantly larger than `bf16-srv` against `bf16-srv-r2`
  (8-bit inside restart noise; H1 restated against a measured floor).
- N2 NVFP4 and Q4_K_M flip rates exceed every non-quantization source, with intervals that exclude them.
- N3 Changing engine at BF16 (`ref` -> `bf16-srv`, `ref` -> `gguf-bf16`) moves at least as many decisions as 8-bit
  quantization within one engine (H7).
- N4 Kernel selection (`ref` vs `ref-fla`) flips a nonzero number of decisions, within the range of the vLLM
  restart drift.
- N5 llama.cpp at BF16 is bit-identical across restarts (`gguf-bf16` vs `gguf-bf16-r2`).

[A closing note on remaining work is omitted.]
