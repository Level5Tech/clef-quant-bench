"""Jev/SystemOne-compatible HTTP server for Clef on vLLM, returning FULL-PRECISION logits.

  POST /v1/systemone   Cloudflare's request body. Response = systemone() body (answers rounded to 4 decimals,
                       as Cloudflare's reference does) PLUS "raw": {qid: {"option_ids": [...], "logits": [...]}}
                       with the unrounded float32 logits straight from the joint head. Clients that need full
                       precision read "raw" and softmax themselves (runners/http_systemone.py does).
  GET  /v1/engine      engine / version / dtype info for result metadata
  GET  /v1/models, /health

  python -m clef_bench_vllm.server --model <release-or-quant-dir> --port 18950

Adapted from the Apache-2.0 server in kurcontko/clef-NVFP4 @ c37a38d (third_party/kurcontko-clef-vllm).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer
from vllm import PoolingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from clef_bench_vllm import register
from clef_bench_vllm.schema import ARCHITECTURE, MEM_STATS_PATH, TASK, schema_layout


def read_memory_stats() -> dict:
    """GPU memory numbers written by the plugin's engine-core process (same container, shared /tmp)."""
    try:
        stats = json.loads(Path(MEM_STATS_PATH).read_text())
    except (OSError, ValueError):
        return {}
    weights = stats["weights_gib"] * 2**30 / 1e9
    activation = stats["activation_peak_gib"] * 2**30 / 1e9
    return {"peak_memory_gb": round(weights + activation, 3), "weights_memory_gb": round(weights, 3),
            "activation_peak_gb": round(activation, 3),
            "memory_note": "peak = GPU weights after load + peak activation during requests (torch allocator); "
                           "excludes the preallocated KV/state cache pool (set by --gpu-memory-utilization)"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local dir: Cloudflare release snapshot or a quantized build")
    ap.add_argument("--served-model-name", default="clef")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=16)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.35,
                    help="fraction of the 128 GB unified memory vLLM may claim (weights + KV/state cache)")
    ap.add_argument("--enforce-eager", action="store_true", help="skip CUDA graph / torch.compile (slow, fewer moving parts)")
    ap.add_argument("--no-warmup", action="store_true")
    return ap.parse_args(argv)


def _release_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", model_dir / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_app(args) -> FastAPI:
    register()
    model_dir = Path(args.model)
    release = _release_module(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    quant_cfg = json.loads((model_dir / "config.json").read_text()).get("quantization_config") or {}
    quant_summary = {"quant_method": quant_cfg.get("quant_method"), "format": quant_cfg.get("format")} if quant_cfg else None
    state: dict = {"engine": None, "ready": False}

    async def score(encoded):
        params = PoolingParams(task=TASK, extra_kwargs={"clef_questions": schema_layout(encoded)})
        final = None
        async for output in state["engine"].encode(
            {"prompt_token_ids": list(encoded.input_ids)}, params, f"so-{uuid.uuid4().hex}"
        ):
            final = output
        if final is None:
            raise HTTPException(500, "engine returned no output")
        return final.outputs.data.float()

    async def warmup() -> None:
        question = {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "First.", "b": "Second."}}
        for words, concurrent in ((50, 1), (400, 4), (1500, 4)):
            record = {"state": "warm up " * words, "questions": {"q": question, "n": {"type": "noul"}}}
            encoded = release.encode_record(tokenizer, record, args.max_model_len)
            await asyncio.gather(*(score(encoded) for _ in range(concurrent)))

    def validate(request: dict) -> None:
        questions = request.get("questions")
        if not isinstance(request.get("model"), str) or "state" not in request:
            raise ValueError("model and state are required")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        for question_id, question in questions.items():
            if not isinstance(question, dict) or question.get("type") not in release.QUESTION_TYPES:
                raise ValueError(f"{question_id}: type must be noul, choice, or score")
            if question["type"] != "noul" and not question.get("criteria"):
                raise ValueError(f"{question_id}: criteria must not be empty")
        if request.get("images") or request.get("videos"):
            raise ValueError("images and videos are not supported (text and JSON state only)")

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["engine"] = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(model_dir),
            hf_overrides={"architectures": [ARCHITECTURE]},
            runner="pooling",
            enable_prefix_caching=False,
            language_model_only=True,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            disable_log_stats=True,
        ))
        if not args.no_warmup:
            await warmup()
        state["ready"] = True
        try:
            yield
        finally:
            state["engine"].shutdown()

    app = FastAPI(title="Clef SystemOne (raw logits)", lifespan=lifespan)

    @app.post("/v1/systemone")
    async def systemone(request: dict):
        t0 = time.perf_counter()
        try:
            validate(request)
            encoded = await asyncio.to_thread(release.encode_record, tokenizer, request, args.max_model_len)
        except (ValueError, TypeError, KeyError) as exc:
            return JSONResponse(status_code=400, content={"error": {"message": str(exc), "type": "invalid_request_error"}})
        t1 = time.perf_counter()
        flat = await score(encoded)
        t2 = time.perf_counter()
        answers, raw, offset = {}, {}, 0
        for question in encoded.questions:
            count = len(question.option_ids)
            logits = flat[offset:offset + count]
            offset += count
            answers[question.question_id] = release.systemone_answer(
                request["questions"][question.question_id],
                dict(zip(question.option_ids, logits.softmax(-1).tolist())))
            raw[question.question_id] = {"option_ids": list(question.option_ids), "logits": logits.tolist()}
        body = {"model": request["model"], "answers": answers, "raw": raw,
                "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0}}
        timing = f"encode;dur={1e3 * (t1 - t0):.2f}, engine;dur={1e3 * (t2 - t1):.2f}"
        return JSONResponse(content=body, headers={"Server-Timing": timing})

    @app.get("/v1/engine")
    async def engine_info():
        import torch
        import transformers
        import vllm
        return {"engine": "vllm+clef_bench_vllm", "engine_version": f"vllm {vllm.__version__}",
                "torch": torch.__version__, "transformers": transformers.__version__,
                "model_dir": str(model_dir), "quantization": quant_summary,
                "head_dtype": os.environ.get("CLEF_HEAD_DTYPE", "bfloat16"), "logits": "raw float32",
                **read_memory_stats()}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": args.served_model_name, "object": "model", "owned_by": "clef"}]}

    @app.get("/health")
    async def health():
        engine = state["engine"]
        if engine is None or engine.errored or not state["ready"]:
            raise HTTPException(503, "engine not ready")
        return {"status": "ok"}

    return app


def main(argv=None) -> None:
    args = parse_args(argv)
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
