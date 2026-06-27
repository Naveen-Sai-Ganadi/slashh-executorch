"""Path B: wrap the AI Hub QNN context binary (teacher_wavlm_broad_npu_int8_qnn.bin
— the fine-tuned, INT8, 100%-Hexagon-NPU model) into an ExecuTorch .pte so the
app's Module.load() can run the ALREADY-COMPILED graph on the S25's NPU. No
re-quantize, no torch.export, no model rebuild.

ExecuTorch's QNN backend supports this via from_context_binary / the qaihub
gen_pte_from_ctx_bin helper: it emits a .pte with a single QNN-delegate node that
points at the prebuilt context bytes.

Runs ONLY on Linux x86_64 inside Docker. This script is discovery-first: it
prints the real installed API (signatures + source) so the wrapper call is exact,
then attempts the wrap. Output: assets/teacher_wavlm_broad_qnn.pte
"""
import inspect
import os
import sys

sys.path.insert(0, os.getcwd())

CTX = "assets/teacher_wavlm_broad_npu_int8_qnn.bin"
OUT = "assets/teacher_wavlm_broad_qnn.pte"
GRAPH = "wavlm"  # op/graph name embedded in the wrapper


def _show(obj, label):
    print(f"\n===== {label} =====", flush=True)
    try:
        print("signature:", inspect.signature(obj), flush=True)
    except Exception as e:  # noqa: BLE001
        print("signature: <n/a>", e, flush=True)
    if obj.__doc__:
        print("doc:", inspect.getdoc(obj), flush=True)
    try:
        print("--- source ---", flush=True)
        print(inspect.getsource(obj), flush=True)
    except Exception as e:  # noqa: BLE001
        print("source: <n/a>", e, flush=True)


def main():
    import executorch
    _loc = getattr(executorch, "__file__", None) or list(getattr(executorch, "__path__", ["?"]))[0]
    print("executorch", getattr(executorch, "__version__", "?"), "at", _loc, flush=True)

    from executorch.backends.qualcomm.serialization.qc_schema import QcomChipset

    # --- 1) discover the wrapping API actually present in this build
    from_ctx = None
    try:
        from executorch.backends.qualcomm.utils.utils import from_context_binary as from_ctx
        _show(from_ctx, "qualcomm.utils.utils.from_context_binary")
    except Exception as e:  # noqa: BLE001
        print("from_context_binary import failed:", e, flush=True)

    gen_pte = None
    for modpath in (
        "executorch.examples.qualcomm.qaihub_scripts.utils.utils",
        "executorch.examples.qualcomm.utils",
    ):
        try:
            mod = __import__(modpath, fromlist=["*"])
            for cand in ("gen_pte_from_ctx_bin", "preprocess_binary", "make_output_dir"):
                fn = getattr(mod, cand, None)
                if fn:
                    _show(fn, f"{modpath}.{cand}")
                    if cand == "gen_pte_from_ctx_bin":
                        gen_pte = fn
        except Exception as e:  # noqa: BLE001
            print(f"{modpath} import failed:", e, flush=True)

    if not os.path.exists(CTX):
        raise SystemExit(f"missing {CTX}")
    ctx_size = os.path.getsize(CTX)
    print(f"\nctx binary: {CTX} ({ctx_size:,} bytes)", flush=True)

    # --- 2) attempt the wrap via from_context_binary (lowest-level, most portable)
    if from_ctx is None:
        raise SystemExit("from_context_binary not available in this executorch build")

    print("\n[wrap] from_context_binary(...) ...", flush=True)
    sig = inspect.signature(from_ctx)
    kwargs = {}
    if "soc_model" in sig.parameters:
        kwargs["soc_model"] = QcomChipset.SM8750
    # positional: (ctx_path, op_name)
    try:
        result = from_ctx(CTX, GRAPH, **kwargs)
    except TypeError as e:
        print("[wrap] positional call failed, retrying keyword form:", e, flush=True)
        result = from_ctx(ctx_path=CTX, op_name=GRAPH, **kwargs)
    print("[wrap] from_context_binary returned:", type(result), flush=True)
    if isinstance(result, dict):
        print("[wrap] keys:", list(result.keys()), flush=True)

    # --- 3) serialize to .pte. Shape depends on what from_context_binary returns;
    # try the common cases and report exactly what we got so we can finalize.
    buf = None
    obj = result
    if isinstance(result, dict):
        for k in ("edge_program_manager", "exported_program", "program", "graph_module"):
            if k in result:
                obj = result[k]
                print(f"[wrap] using result['{k}'] -> {type(obj)}", flush=True)
                break

    for attr in ("to_executorch",):
        if hasattr(obj, attr):
            prog = getattr(obj, attr)()
            buf = prog.buffer
            print(f"[wrap] serialized via {attr}()", flush=True)
            break
    if buf is None and hasattr(obj, "buffer"):
        buf = obj.buffer

    if buf is None:
        print("[wrap] UNRESOLVED return shape — printing dir() to finalize next pass:", flush=True)
        print(dir(obj), flush=True)
        raise SystemExit("could not serialize; see dir() above")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(buf)
    print(f"[done] wrote {OUT} ({len(buf):,} bytes)", flush=True)
    print("WAVLM_CTX_WRAP_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
