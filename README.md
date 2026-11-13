# The Noise Floor of a Quantization Comparison

Code, results and analysis for the paper of the same title, by Aleks Blagojevich and Trevor Sunderland (Level5). The paper measures
engine, restart, batch and kernel drift against 8-bit and 4-bit quantization effects on two of Cloudflare's decision models, Clef-flash (9B)
and Clef (27B), on one NVIDIA GB10.

Paper: [arXiv:2611.15671](https://arxiv.org/abs/2611.15671)

Archived release (all versions): [10.5281/zenodo.24184497](https://doi.org/10.5281/zenodo.24184497)

## What's here

| Path | Contents |
|---|---|
| `results/` | Per-record outputs for every configuration. IDs, logits and probabilities only, no question text. See `results/README.md` |
| `results-latency/` | Isolated latency runs |
| `analysis/`, `analysis_r3/` | Analysis outputs behind the paper's numbers |
| `bench/` | Analysis code |
| `sweep/`, `runners/`, `serve/`, `quant/` | The sweep driver, engine runners, serving plugin and quantization recipes |
| `datasets/`, `data/` | Dataset builder, record IDs, and the records whose licenses permit redistribution |
| `leakscan/` | The scanner run on this tree before release |
| `PREREGISTRATION.md` | The pre-registered hypotheses and report tests |

Quantized weights are at https://huggingface.co/Level5Tech.

## Reproducing the numbers

Python 3.12. From the repository root:

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python datasets/build.py     # rebuilds all 12 tasks and the calibration set from pinned upstream revisions; needs network access
python -m pytest
python -m bench.analyze --records data/records --results results --out analysis/flash_public --models clef-flash
python -m bench.analyze --records data/records --results results --out analysis/all_public
python -m bench.decompose --out analysis
python -m bench.decompose --model clef --out analysis/clef_decomp
python -m bench.hypotheses
python -m bench.restart_probes --records-dir data/records --results-dir results --model clef --out analysis_r3
```

These commands reproduce the shipped `analysis/` and `analysis_r3/` files exactly. `datasets/build.py` reproduces `data/` byte for byte, including the six
tasks released as IDs.

Three steps can't be rerun from this repository, because they read data that isn't in it:

- `python -m bench.analyze --private private` and `python -m bench.private_agg` need the private sets. Their aggregate output is in
  `analysis/private_aggregates.*`.
- `python -m bench.findings` also needs the project's plan file and the raw private analysis. Its output is `analysis/findings.md`.
- `python -m bench.speed` runs, but it omits the private sets. The shipped `analysis/speed.*` includes them and predates the third-instance runs.

## Data

Six tasks are released as records: BANKING77, CLINC150, MMLU, ARC-Challenge, BoolQ and When2Call. The other six (ANLI R1 to R3, ContractNLI, RAGTruth and SST-5)
are released as IDs, and `datasets/build.py` rebuilds them. `datasets/DATASETS.md` lists sources, pinned revisions and licenses.

The private operational sets aren't released. They appear only as aggregates, under generic set and question names.

## License

Code is under Apache-2.0 (`LICENSE`). Results and analysis outputs are under CC BY 4.0. `third_party/` keeps its own license and attribution. Datasets keep their
own licenses, listed in `datasets/DATASETS.md`.

## Citation

Please cite the paper:

```bibtex
@misc{blagojevich2026noisefloorquantizationcomparison,
  title={The Noise Floor of a Quantization Comparison: Engine, restart, batch and kernel drift, measured against 8-bit and 4-bit effects in two decision models},
  author={Aleks Blagojevich and Trevor Sunderland},
  year={2026},
  eprint={2611.15671},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2611.15671},
}
```

`CITATION.cff` holds the same entry, and GitHub's "Cite this repository" button reads it.
