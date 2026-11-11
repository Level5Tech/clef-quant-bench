# Attribution

Vendored, unmodified, for reference and attribution only (nothing here is imported at runtime):

- Source: https://huggingface.co/kurcontko/clef-NVFP4 (unofficial community quantization of Cloudflare/clef)
- Revision fetched: `c37a38dc58929e81665af7a5c4c93605b1890ed2` (lastModified 2026-10-07T21:19:29Z), fetched 2026-10-15
- License: Apache-2.0 (file `LICENSE` in this directory is the copy shipped in that repo)
- Files: `vllm_plugin/` (clef_vllm plugin + server, built for vLLM 0.28.x), `scripts/{build_data,quantize,shrink_tables}.py`, `recipe.yaml`

Derived work (Apache-2.0, modified): `serve/plugin/clef_bench_vllm/model.py` and `server.py` adapt the plugin's pooler
design (collect each request's final hidden states across chunked prefill, run the release's JointSchemaHead on the
full prompt, per-request schema layout in `PoolingParams.extra_kwargs`) to vLLM 0.29, drop the int8 lookup tables, and
return raw logits. `quant/recipe.py` credits the fused in_proj_qkv/in_proj_z NVFP4 global-scale finding to this repo's
`scripts/quantize.py`; the quantization scripts in `quant/` were written independently.
The original joint-schema code (`joint_schema_model.py`) is Cloudflare's, Apache-2.0, loaded unmodified from the release.
