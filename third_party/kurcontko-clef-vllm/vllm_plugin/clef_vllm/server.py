"""Jev/SystemOne-compatible HTTP API for Clef models on vLLM.

  POST /v1/systemone   same request/response bodies as joint_schema_model.systemone()
  GET  /v1/models, /health

Requests are encoded with the release's encode_record and scored by vLLM, which batches
concurrent requests continuously. --dp N runs one engine per GPU behind one port.

  clef-systemone --model ./clef-flash-NVFP4 --dp 2 --port 8000
"""
import argparse
import asyncio
import importlib.util
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

from clef_vllm.schema import ARCHITECTURE, TASK, schema_layout


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local path of a Clef checkpoint (snapshot directory)")
    ap.add_argument("--served-model-name", default="clef-flash")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--dp", type=int, default=1, help="data-parallel engines, one per GPU")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel GPUs per engine")
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--flashinfer-autotune", choices=["auto", "on", "off"], default="auto",
                    help="FlashInfer kernel autotuning; auto = off when --tp > 1 (vLLM 0.28 can deadlock "
                         "autotuning NVFP4 GEMMs across tensor-parallel ranks)")
    ap.add_argument("--max-cudagraph-tokens", type=int, default=2048,
                    help="capture CUDA graphs for engine steps up to this many tokens (0 = vLLM default, which "
                         "stops at 2 x max-num-seqs tokens and leaves every longer prefill step on the slow eager path)")
    ap.add_argument("--gdn-backend", choices=["auto", "flashinfer", "cutedsl", "triton"], default="auto",
                    help="force Qwen3.5's linear-attention prefill kernel. vLLM 0.28 only picks FlashInfer on SM 9.0/10.x; "
                         "'flashinfer' also works on RTX 50-series (SM 12.0) and was ~10%% faster there (experimental)")
    ap.add_argument("--no-warmup", action="store_true",
                    help="skip the startup warm-up (first requests then pay kernel JIT/autotune time)")
    return ap.parse_args(argv)


def _release_module(model_dir: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", model_dir / "joint_schema_model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def create_app(args) -> FastAPI:
    if args.gdn_backend != "auto":
        os.environ["CLEF_GDN_BACKEND"] = args.gdn_backend  # read by clef_vllm.model in the engine processes
    model_dir = Path(args.model)
    release = _release_module(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    app = FastAPI(title="Clef SystemOne API")
    state = {"engine": None, "ready": False}

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
        """Run batches of several sizes so kernel JIT/autotuning happens before serving."""
        question = {"type": "choice", "instructions": "Pick one.", "criteria": {"a": "First.", "b": "Second."}}
        for words, concurrent in ((50, 1), (400, 8), (1500, 32), (6000, 4), (200, args.max_num_seqs)):
            record = {"state": "warm up " * words, "questions": {"q": question, "n": {"type": "noul"}}}
            encoded = release.encode_record(tokenizer, record, args.max_model_len)
            await asyncio.gather(*(score(encoded) for _ in range(concurrent)))

    def validate(request: dict) -> None:
        """The checks of joint_schema_model.systemone(), plus text-only input."""
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
            raise ValueError("images and videos are not supported by this server (text and JSON state only)")

    if args.max_cudagraph_tokens > 0:
        # Clef is prefill-only: a step holds whole prompts, so graphs must cover prompt-sized token counts.
        limit = min(args.max_cudagraph_tokens, args.max_num_batched_tokens)
        sizes = sorted({1, 2, 4, 8, 16, *range(32, min(limit, 1024) + 1, 32), *range(1024, min(limit, 4096) + 1, 128),
                        *range(4096, limit + 1, 512)})
        graph_kwargs = {"compilation_config": {"cudagraph_capture_sizes": sizes}}
    else:
        graph_kwargs = {}
    autotune = args.flashinfer_autotune == "on" or (args.flashinfer_autotune == "auto" and args.tp == 1)
    kernel_kwargs = {} if autotune else {"kernel_config": {"enable_flashinfer_autotune": False}}

    @app.on_event("startup")
    async def start_engine() -> None:
        state["engine"] = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=str(model_dir),
            hf_overrides={"architectures": [ARCHITECTURE]},
            runner="pooling",
            tensor_parallel_size=args.tp,
            data_parallel_size=args.dp,
            enable_prefix_caching=False,
            language_model_only=True,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_log_stats=True,
            **kernel_kwargs,
            **graph_kwargs,
        ))
        if not args.no_warmup:
            await warmup()
        state["ready"] = True

    @app.on_event("shutdown")
    async def stop_engine() -> None:
        if state["engine"] is not None:
            state["engine"].shutdown()

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
        answers, offset = {}, 0
        for question in encoded.questions:
            count = len(question.option_ids)
            probabilities = flat[offset:offset + count].softmax(-1).tolist()
            offset += count
            answers[question.question_id] = release.systemone_answer(
                request["questions"][question.question_id], dict(zip(question.option_ids, probabilities)))
        body = {
            "model": request["model"],
            "answers": answers,
            "usage": {"input_tokens": len(encoded.input_ids), "output_tokens": 0},
        }
        t3 = time.perf_counter()
        timing = f"encode;dur={1e3 * (t1 - t0):.2f}, engine;dur={1e3 * (t2 - t1):.2f}, answer;dur={1e3 * (t3 - t2):.2f}"
        return JSONResponse(content=body, headers={"Server-Timing": timing})

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
