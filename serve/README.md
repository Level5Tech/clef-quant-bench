# GPU side: runners, quantization, serving (Spark)

All commands run ON the Spark (`ssh spark`), from `~/clef-quant/bench`. Sync from the Mac with `serve/sync_to_spark.sh`
(pushes only runners/, quant/, serve/, third_party/, data/smoke/, PLAN.md). Containers are named `clefbench-*`.
Ports used here: 18951 bf16, 18952 fp8, 18953 nvfp4, 18954 llama.cpp (stay clear of 8001 8002 8004 9292 11434 18770 8770 8771 3000).
GPU RULE: `clef` (27B) commands are started by the orchestrator only, one job at a time.

## 1. BF16 reference (transformers, Cloudflare's load_release_model)
    runners/run_native_ref.sh clef-flash --records /bench/data/records/*.jsonl --out-dir /results/clef-flash/ref
    runners/run_native_ref.sh clef       --records /bench/data/records/*.jsonl --out-dir /results/clef/ref     # 27B
    # throughput mode: add  --batch-size 8 --config ref-b8  and a different --out-dir
Writes `<out-dir>/<task>.jsonl`, `<task>.jsonl.meta.json`, and `<out-dir>/meta.json`.

## 2. Quantize (llm-compressor 0.14.0 in ~/clef-quant/venv-quant; one-time `quant/setup_quant_env.sh`)
    quant/build_quants.sh clef-flash fp8
    quant/build_quants.sh clef-flash nvfp4 --calibration /bench/data/calibration.jsonl --num-samples 512
    quant/build_quants.sh clef fp8                                                                           # 27B
    quant/build_quants.sh clef nvfp4 --calibration /bench/data/calibration.jsonl --num-samples 512           # 27B
Outputs `~/clef-quant/models/<model>-<fmt>/` (+ `clef_quant_recipe.json`, head in `clef_head/`). Each build ends with
`quant/verify_checkpoint.py`. Ignore list: `quant/recipe.py`; module inventory: `quant/run_quant.sh quant.list_linear_modules <release>`.

## 3. Serve through vLLM 0.29 + clef_bench_vllm plugin (raw logits at POST /v1/systemone, field "raw")
    serve/serve_clef.sh clef-flash bf16  18951 --gpu-memory-utilization 0.25
    serve/serve_clef.sh clef-flash fp8   18952 --gpu-memory-utilization 0.25
    serve/serve_clef.sh clef-flash nvfp4 18953 --gpu-memory-utilization 0.25
    serve/serve_clef.sh clef bf16  <port> --gpu-memory-utilization 0.55                                      # 27B, NOT RUN (54 GB weights)
    serve/serve_clef.sh clef fp8   <port> --gpu-memory-utilization 0.40                                      # 27B, NOT RUN
    serve/serve_clef.sh clef nvfp4 <port> --gpu-memory-utilization 0.30                                      # 27B, NOT RUN
    docker stop clefbench-serve-<model>-<fmt>
Foreground process: background it with `nohup ... &` and wait for `curl :<port>/health`. Start takes ~2.5 min (torch.compile + CUDA graphs).

## 4. llama.cpp (GGUF) — native process
    serve/serve_llamacpp.sh clef-flash Q8_0 18954       # needs -b/-ub 16384 (built in); kill $(cat ~/clef-quant/logs/llama-clef-flash-Q8_0.pid)

## 5. Client for either server
    python3 -m runners.http_systemone --url http://127.0.0.1:<port> --records /bench/data/records/*.jsonl \
        --out-dir ~/clef-quant/results/<model>/<config> --model-name <model> --config <config> \
        --model-dir <weights dir> | --model-file <gguf>   [--server-pid <llama-server pid>]
    python3 -m runners.compare A.jsonl B.jsonl ;  python3 -m runners.compare_dirs results/<m>/ref results/<m>/<config>
Writes `<out-dir>/<task>.jsonl` and `<out-dir>/meta.json` {file_size_gb, peak_memory_gb, engine, engine_version, ...}.
