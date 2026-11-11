# sweep/: resumable queue for the full Clef / Clef-flash measurement sweeps

Runs ON the Spark (`~/clef-quant/bench/sweep/`). Authored on the Mac, synced with
`rsync -a --exclude __pycache__ sweep/ spark:~/clef-quant/bench/sweep/` (own directory only, never `--delete`).

## Pieces
| file | role |
|---|---|
| `configs.py` | the 8 quality configs, ports, `--gpu-memory-utilization` ladders, worker caps (all 1, see "Concurrency") |
| `jobs.py` | writes `jobs.jsonl`: one line per (model, config, set) plus one latency line per config and one build line per 27B quant; `stages()` interprets it |
| `driver.py` | entry point: four stages, locking, STOP file, latency sessions, skip rules |
| `sched.py` | quality scheduler: units, workers, merge in record order, per-set sidecars, `meta.json` |
| `units.py` | pure helpers: shard ranges, atomic claims, ordered merge, memory model, launch rule |
| `engines.py` | native worker (container), vLLM / llama.cpp servers, HTTP client per unit, quant builds, GPU-idle gate |
| `native_worker.py` | in-container reference worker: same forward as `runners/native_ref.py`, claims units from a plan |
| `in_container.sh` | `docker run` of a python module with identity bind-mounts (private records in, private results out); `CLEF_PYTHONPATH_EXTRA` is appended to the container `PYTHONPATH` (the fla-kernel worker's flash-linear-attention install, `~/clef-quant/pylibs-fla`, which the NGC image lacks) |
| `swap_driver.sh` | on the Spark: STOP at the next unit boundary, sync `bench-staged/` -> `bench/`, optional post-sync check (no relaunch if it fails), relaunch the driver, watchdog paused meanwhile, one Signal message |
| `check_fla_container.sh` | on the Spark: proves the NGC container imports flash-linear-attention from `~/clef-quant/pylibs-fla` through `docker_env.sh`'s `CLEF_PYTHONPATH_EXTRA` path; the post-sync check for fla-related swaps |
| `latency/` | the separate latency benchmark (`latency_http.py`, `latency_native.py`, `stats.py`) |
| `mps.py` | optional NVIDIA MPS daemon (private pipe dir). Measured: no gain here, not used by the driver |
| `proof.py` | determinism proofs and the concurrency benchmark (public data only) |
| `status.py` | progress and projected finish |

## Stages (order of the brief)
1. Clef-flash quality, configs `ref, q8_0, q4_k_m, fp8, nvfp4, bf16-srv, gguf-bf16, ref-r2`.
2. Build Clef 27B `fp8` and `nvfp4` (`quant/build_quants.sh`, nvfp4 with `--calibration data/calibration.jsonl --num-samples 512`,
   each build ends with `verify_checkpoint`). Alone on the GPU.
3. Clef 27B quality, same eight configs.
4. **Latency phase** for every config of both models, one session at a time on an otherwise idle GPU
   (the driver waits until no other `clefbench-*` job or clef-quant llama-server runs). Latency never shares the GPU.

Records per job: all 12 public files from `~/clef-quant/bench/data/records` and the 4 private sets from
`~/clef-quant/private/records`. Outputs: public `~/clef-quant/results/<model>/<config>/<set>.jsonl` + `meta.json`;
private ONLY `~/clef-quant/private/results/<model>/<config>/<set>.jsonl` + `meta.json` (directories 700, files hardened
go-rwx); latency `~/clef-quant/results-latency/<model>/<config>.json`. Shards live in `~/clef-quant/sweep-work` (public)
and `~/clef-quant/private/sweep-work` (private, 700) and are deleted after the merged set is verified. Private-set
server and worker logs are in `~/clef-quant/logs/sweep/` (700); `sweep.log` carries counts, timings and exit codes only.

## Units, resume, verification
A set is cut into **units** (contiguous record ranges of ~10 minutes of work, starts aligned to multiples of 8 so that
`ref-r2`'s batches of 8 are identical to an unsharded run). A worker (one model copy) claims units atomically
(`mkdir`), writes each to a `.part` file and renames it when complete. When all units of a set exist they are merged in
record order, ids checked against the records file, and the final file is written under a temporary name and renamed.
A set counts as done only if its final file has the records file's line count AND every line is valid JSON.
Restart the same command after any stop or crash: finished sets are skipped, an interrupted unit is re-run (at most
~10 minutes lost). A leftover partial file at a final path is never trusted; it is replaced by the merge.

## Configs
| config | engine | notes |
|---|---|---|
| `ref` | transformers, Cloudflare `load_release_model`, BF16, batch 1 | ground truth |
| `ref-r2` | same, **`--batch-size 8`** | **the realistic serving-noise floor.** Not a bit-for-bit repeat of `ref`: batching pads sequences and changes kernel shapes, so the difference to `ref` is the numerical noise a production service that batches requests would see anyway. Any quantization effect smaller than this is reported as noise. (A bs=1 repeat of `ref` is exactly 0 on this path and says nothing.) |
| `bf16-srv`, `fp8`, `nvfp4` | vLLM 0.29 + `clef_bench_vllm`, `--max-num-seqs 1`, client sends one request at a time | ports 18951 / 18952 / 18953 |
| `gguf-bf16`, `q8_0`, `q4_k_m` | llama.cpp `llama-server`, `-b 16384 -ub 16384 -c 16384` via `serve/serve_llamacpp.sh` | port 18954 |

`--gpu-memory-utilization` is tried from a ladder (`configs.GPU_UTIL_QUALITY`); the value that brought the server up is
logged (`SERVER_UP ... gpu_memory_utilization=`). Clef-flash quality servers use a lean 0.21 / 0.16 / 0.13
(bf16 / fp8 / nvfp4) because `--max-num-seqs 1` needs almost no KV cache; the latency phase uses the README values
(0.25). The 27B values are guesses with fallbacks.

## Concurrency: measured, and why the queue runs serial
The brief asked whether the Spark can go faster by running several batch-size-1 jobs at once. Measured on this machine
(Clef-flash, same fixed workload per worker, start barrier, nvidia-smi sampled every 0.5 s):

| engine | workload | N=1 | N=2 | speedup N=2 |
|---|---|---|---|---|
| native BF16 | mixed (compute- and floor-bound sets) | 1.99 rec/s | 1.80 rec/s | 0.90x |
| native BF16 + MPS | mixed | 2.00 rec/s | 1.81 rec/s | 0.90x |
| native BF16 | short-input sets only | 6.33 rec/s | 5.69 rec/s | 0.90x |
| llama.cpp q8_0 | short-input sets only | 8.29 rec/s | 7.77 rec/s | 0.94x |

No configuration got faster with a second worker; GPU power stays 47-55 W for native at every N. The native runner is
**not** launch-bound: at batch size 1 each record streams the whole 19 GB of BF16 weights through the GB10's memory
(about 70 ms at its ~273 GB/s, against a measured ~139 ms floor for short inputs), and long inputs are compute-bound.
The busy CPU core is the process spinning on CUDA sync. Extra workers share the same memory bandwidth and SMs, and
cost 12 GB of host memory each. Throughput therefore comes only from batching (which changes numerics: that is `ref-r2`)
or from a smaller model file (fp8/nvfp4/GGUF, already faster: 15 rec/s for vLLM fp8).

**Safety lesson, measured the hard way:** a native worker holds ~12 GB of host memory on top of its 19 GB of GPU-resident
weights (unified memory: both count against the same 121 GB). Four at once exhausted the machine and the kernel OOM
killer took down other services (see `units.native_mem_gb`, `proof.mem_guard`). `configs.WORKER_CAPS` and
`GLOBAL_SLOTS` are 1; the scheduler also enforces a memory budget (MemAvailable minus 12 GB headroom) and starts workers
one at a time, so raising a cap cannot repeat that.

## Relaunching over finished configs

A relaunch re-enters `finalize()` for every config, including finished ones, and worker counts live in the process. So
`_write_config_meta` leaves an existing `meta.json` alone unless this process ran a worker or merged a set for it. When it
does write, it keeps the larger of the recorded and current worker counts and the union of the GPU memory settings. If
`meta.json` is missing it is rebuilt from the sidecars with `concurrent_workers_max: null`, because this process saw no
worker. Before this guard, relaunches on 2026-10-18 and 2026-11-03 overwrote the value with 0; see `results/README.md`.

## Determinism proofs (concurrency does not change numerics)
Public data, Clef-flash, GPU shared with 2 other busy workers during the "concurrent" runs:
* native: `results/clef-flash/ref/anli_r1.jsonl` (1000 records, written by the serial single-worker run in an earlier
  container) re-run next to 2 busy workers: **max |delta logit| = 0.0**, 3000 logits, 0 id/option/token mismatches.
* llama.cpp q8_0: 200 records, serial vs next to 2 busy workers: **max |delta prob| = 0.0** (llama.cpp reports probabilities,
  no logits), 0 rows differ.
* vLLM fp8: **identical within one server instance** (same 200 records alone, next to 2 busy workers, alone again:
  max |delta logit| = 0.0). **Across server instances vLLM is NOT reproducible:** two serial runs of the same flags on a
  fresh server differ (max |delta logit| 0.94, 199/200 rows), as do the README flags vs `--max-num-seqs 1`. So a
  vLLM result file is one server instance's numerics; the driver therefore uses exactly one server per vLLM config, and a
  resume after a crash starts a new instance (visible in `sweep.log` as another `SERVER_UP`). This instance-to-instance
  spread is part of the vLLM noise floor and should be reported with it.
Raw reports: `~/clef-quant/sweep-proof/report/*.json` on the Spark.

## Latency benchmark
Per config, in the isolated final phase: fixed sample of 300 records per public task and 300 per private set (seeded,
identical for every config; sets with fewer records use all), batch size 1, 20 warm-up requests (round-robin over all sets,
untimed), then every sampled request timed. Output: p50/p95/p99/mean per set and overall, decisions/sec, input-token
buckets with their own percentiles, peak memory. Private sets appear as per-set aggregates only (no ids, text or errors).
* Timing window: native = encode + collate + forward + cuda sync in-process; HTTP engines = localhost round trip including
  server-side tokenisation. Compare engines with that difference in mind.
* Peak memory: vLLM = plugin weights + peak activation (read after the run) and nvidia-smi over the container's processes;
  llama.cpp = nvidia-smi of the server process; native = `torch.cuda.max_memory_allocated`.
* `ref-r2` latency is an alias of `ref` (`alias_of`): latency is defined at batch size 1.
* The per-row `latency_ms` in the quality outputs was measured while the queue ran other jobs of this driver
  (serial now, but the machine also runs other services): use `results-latency/` for latency numbers.

## Run / control
```
cd ~/clef-quant/bench
python3 -m sweep.jobs --records-dir data/records --out sweep/jobs.jsonl
nohup setsid python3 -m sweep.driver run --jobs sweep/jobs.jsonl >> ~/clef-quant/logs/sweep-driver.out 2>&1 < /dev/null &
python3 -m sweep.status --jobs sweep/jobs.jsonl          # progress + projected finish
touch ~/clef-quant/sweep-state/STOP                      # stop after the current units
```
A second driver instance exits immediately (flock on `sweep-state/driver.lock`). On start the driver stops only the
containers/processes an earlier instance recorded in `sweep-state/active.json`.
`sweep.log` events: `STAGE_START`, `WORKER_START`, `SERVER_UP`, `END job= exit= records= errors= unit_secs= elapsed=`,
`CONFIG_DONE`, `FAIL`, `SKIP`, `STATUS`, `SESSION_START/END` (latency), `DRIVER_END`.

## Tests
`pytest sweep/tests runners/tests` on the Mac (stats, job order, units, claims, merge, scheduler with fake workers,
failure -> abandon, STOP, memory guard, exact comparison).
