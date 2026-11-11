"""runners.gdn_kernels: explicit, loud kernel choice for the native reference path (no torch/transformers needed)."""
import sys
import types

import pytest

from runners import gdn_kernels


def _fake_fla(monkeypatch, with_chunk=True):
    fla = types.ModuleType("fla")
    fla.__version__ = "9.9.9"
    ops = types.ModuleType("fla.ops")
    gdr = types.ModuleType("fla.ops.gated_delta_rule")
    calls = []

    def chunk_gated_delta_rule(*a, **k):
        calls.append(k)
        return "fla-out"

    if with_chunk:
        gdr.chunk_gated_delta_rule = chunk_gated_delta_rule
        ops.chunk_gated_delta_rule = chunk_gated_delta_rule
    fla.ops = ops
    ops.gated_delta_rule = gdr
    for name, m in (("fla", fla), ("fla.ops", ops), ("fla.ops.gated_delta_rule", gdr)):
        monkeypatch.setitem(sys.modules, name, m)
    return calls


def _fake_modeling(monkeypatch, bind_from_fla: bool):
    """Mimics transformers' decorator: AT IMPORT it captures fla's function if `fla` imports, else its own.

    Built lazily, when gdn_kernels imports the module, because that ordering is the whole point.
    """
    holder = {}

    def torch_impl(*a, **k):
        return "torch-out"

    def build():
        mod = types.ModuleType("fake_modeling")
        try:
            import fla  # noqa: F401
            impl = sys.modules["fla.ops"].chunk_gated_delta_rule if bind_from_fla else torch_impl
        except ImportError:
            impl = torch_impl

        def wrapped(*a, **k):
            return impl(*a, **k)

        mod.torch_chunk_gated_delta_rule = wrapped
        holder["mod"] = mod
        return mod

    monkeypatch.setattr(gdn_kernels, "_import_modeling", build)
    monkeypatch.setattr(gdn_kernels, "MODULE", "fake_modeling_not_in_sys_modules")
    return holder


def test_torch_blocks_fla_and_returns_descriptor(monkeypatch):
    _fake_fla(monkeypatch)
    assert gdn_kernels.install("torch") == gdn_kernels.TORCH_DESCRIPTOR
    assert sys.modules["fla"] is None
    with pytest.raises(ImportError):
        import fla  # noqa: F401
    gdn_kernels.assert_used("torch")   # never raises for torch
    monkeypatch.delitem(sys.modules, "fla")


def test_fla_binds_counts_and_asserts(monkeypatch):
    calls = _fake_fla(monkeypatch)
    holder = _fake_modeling(monkeypatch, bind_from_fla=True)
    assert gdn_kernels.install("fla") == "fla-9.9.9"
    with pytest.raises(RuntimeError, match="never called"):
        gdn_kernels.assert_used("fla")
    assert holder["mod"].torch_chunk_gated_delta_rule(1, 2, cu_seqlens=None) == "fla-out"
    assert calls == [{"cu_seqlens": None}]
    assert gdn_kernels.calls() == 1
    gdn_kernels.assert_used("fla")


def test_fla_missing_is_loud(monkeypatch):
    for name in list(sys.modules):
        if name == "fla" or name.startswith("fla."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setattr(gdn_kernels, "MODULE", "fake_modeling_not_in_sys_modules")
    monkeypatch.setattr(sys, "path", [p for p in sys.path])   # fla is not installed in the test env
    with pytest.raises(ImportError):
        gdn_kernels.install("fla")


def test_fla_not_bound_is_loud(monkeypatch):
    _fake_fla(monkeypatch)
    _fake_modeling(monkeypatch, bind_from_fla=False)
    with pytest.raises(RuntimeError, match="did not bind"):
        gdn_kernels.install("fla")


def test_too_late_and_bad_kind(monkeypatch):
    monkeypatch.setattr(gdn_kernels, "MODULE", "sys")   # something already imported
    with pytest.raises(RuntimeError, match="already imported"):
        gdn_kernels.install("torch")
    monkeypatch.setattr(gdn_kernels, "MODULE", "fake_modeling_not_in_sys_modules")
    with pytest.raises(ValueError):
        gdn_kernels.install("cuda")
