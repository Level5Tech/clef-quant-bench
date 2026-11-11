"""Client-side constants and helpers. No vLLM imports, so clients and tests can use them cheaply."""
from __future__ import annotations

ARCHITECTURE = "ClefBenchForDecision"
TASK = "token_classify"
MEM_STATS_PATH = "/tmp/clef_bench_mem_stats.json"  # written by the engine-core process, read by the server


def schema_layout(encoded) -> list:
    """PoolingParams.extra_kwargs["clef_questions"] for an EncodedRecord from joint_schema_model.encode_record.

    One entry per question: [question_type_id, [q_start, q_end], [[opt_start, opt_end], ...]].
    """
    return [[q.question_type, list(q.question_span), [list(s) for s in q.option_spans]] for q in encoded.questions]
