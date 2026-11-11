"""Kernel choice for the Gated DeltaNet layers of the transformers (native) reference path.

transformers 5.x decorates Qwen3.5's `torch_chunk_gated_delta_rule` (24 of Clef-flash's 32 layers route through it)
with `use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")`. At import of the modeling module the
decorator tries, in order: a `kernels` hub build, the `fla` package (flash-linear-attention), its own pure-PyTorch
implementation. The benchmark container is offline and shipped neither `kernels` nor `fla`, so the pure-PyTorch path
ran: thousands of tiny kernels per record, launch-bound (2.0 records/s on Clef-flash on the GB10, measured 2026-10-16).

The binding is fixed at import time and captured by the layer class, so it cannot be patched afterwards. This module
therefore makes the choice explicit BEFORE the modeling module is imported:
  * install("fla"): fla must import (else ImportError), its entry points get counting wrappers, the modeling module
    is imported and its decorator closure is checked to hold the fla kernel; assert_used() after warm-up checks the
    kernel actually ran. A run can never claim "fla" while the fallback did the work.
  * install("torch"): the `fla` import is blocked for this process (sys.modules["fla"] = None), so the decorator
    takes the pure-PyTorch path even when fla is installed. The reference (`ref`, `ref-r2`) runs this way.
Measured 2026-10-16 (Clef-flash, 200 anli_r1 records, shared GPU): the torch path reproduces the sweep's reference
bit for bit across processes; fla is 1.48x faster and differs from it by max |dlogit| 0.5625, max |dprob| 0.040,
1 argmax flip. That drift is measured by the `ref-fla` config rather than adopted into the reference.
Every native result records the descriptor in its meta (`gdn_kernel`) and in `engine_version`.
"""
from __future__ import annotations

import importlib
import sys

MODULE = "transformers.models.qwen3_5.modeling_qwen3_5"
NAME = "torch_chunk_gated_delta_rule"
FLA_ENTRY_POINTS = ("chunk_gated_delta_rule", "fused_recurrent_gated_delta_rule")
KINDS = ("torch", "fla")
TORCH_DESCRIPTOR = "transformers-torch-fallback"
_calls = 0
_wrappers: list = []


def _import_modeling():
    return importlib.import_module(MODULE)


def _wrap_counting(owner, name) -> None:
    orig = getattr(owner, name)

    def counting(*args, **kwargs):
        global _calls
        _calls += 1
        return orig(*args, **kwargs)

    counting.__wrapped__ = orig
    counting.__name__ = name
    setattr(owner, name, counting)
    _wrappers.append(counting)


def _closure_callables(func) -> list:
    out = []
    for cell in getattr(func, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if callable(value):
            out.append(value)
    return out


def install(kind: str) -> str:
    """Bind the GDN kernel for this process; call before anything imports the modeling module.

    Returns the descriptor written to result meta.
    """
    global _calls
    if kind not in KINDS:
        raise ValueError(f"gdn kernel must be one of {KINDS}, got {kind!r}")
    if MODULE in sys.modules:
        raise RuntimeError(f"{MODULE} is already imported; the kernel binding is fixed at import and cannot be changed now")
    _calls = 0
    _wrappers.clear()
    if kind == "torch":
        sys.modules["fla"] = None  # type: ignore[assignment]  # makes `import fla` raise: transformers takes its torch path
        return TORCH_DESCRIPTOR
    if sys.modules.get("fla", True) is None:
        del sys.modules["fla"]
    import fla  # noqa: PLC0415 - optional dependency, imported only when asked for
    import fla.ops  # noqa: PLC0415
    import fla.ops.gated_delta_rule  # noqa: PLC0415

    for owner in (fla.ops, fla.ops.gated_delta_rule):
        for name in FLA_ENTRY_POINTS:
            if hasattr(owner, name):
                _wrap_counting(owner, name)
    mod = _import_modeling()
    bound = getattr(mod, NAME)
    if not any(c in _wrappers for c in _closure_callables(bound)):
        raise RuntimeError(f"{MODULE}.{NAME} did not bind to fla's kernel (closure: "
                           f"{[getattr(c, '__qualname__', repr(c)) for c in _closure_callables(bound)]})")
    return f"fla-{fla.__version__}"


def calls() -> int:
    return _calls


def assert_used(kind: str) -> None:
    """After at least one forward pass: the fla kernel must have run, or the "fla" label would be a lie."""
    if kind == "fla" and _calls == 0:
        raise RuntimeError(f"gdn kernel 'fla' is bound but was never called by {MODULE}.{NAME}")
