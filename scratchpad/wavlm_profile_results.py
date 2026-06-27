"""Fetch WavLM INT8 Hexagon profile results (job jglo88klg): wait for completion,
then summarize on-device latency + per-compute-unit (NPU/GPU/CPU) layer offload."""
import collections
import time
from pathlib import Path

import qai_hub as hub

PROFILE = "jglo88klg"
OUT = Path("scratchpad/jglo88klg_profile")


def main():
    job = hub.get_job(PROFILE)
    print(f"profile job {PROFILE}: waiting …", flush=True)
    t0 = time.time()
    while True:
        st = job.get_status()
        if not st.running:
            break
        time.sleep(15)
    st = job.get_status()
    print(f"status: code={st.code} success={st.success} ({time.time()-t0:.0f}s wait)", flush=True)
    if not st.success:
        print(f"PROFILE FAILED: {st.message}", flush=True)
        return

    prof = job.download_profile()  # dict
    ex = prof.get("execution_summary", {})
    est = ex.get("estimated_inference_time")  # microseconds
    peak = ex.get("inference_memory_peak_range")
    print(f"\n=== ON-DEVICE LATENCY (Samsung Galaxy S24, Hexagon) ===", flush=True)
    if est is not None:
        print(f"  estimated inference time: {est/1000:.2f} ms ({est} us)", flush=True)
    print(f"  peak memory range: {peak}", flush=True)

    # per-layer compute-unit breakdown
    details = prof.get("execution_detail", []) or prof.get("layer_details", [])
    units = collections.Counter()
    unit_time = collections.Counter()
    for d in details:
        cu = d.get("compute_unit", "?")
        units[cu] += 1
        t = d.get("execution_time", 0) or 0
        unit_time[cu] += t
    total_layers = sum(units.values())
    total_time = sum(unit_time.values()) or 1
    print(f"\n=== PER-LAYER COMPUTE-UNIT OFFLOAD ({total_layers} layers) ===", flush=True)
    for cu in sorted(units, key=lambda k: -units[k]):
        print(f"  {cu:5s}: {units[cu]:4d} layers ({100*units[cu]/total_layers:5.1f}%)  "
              f"time {100*unit_time[cu]/total_time:5.1f}%", flush=True)

    OUT.mkdir(exist_ok=True)
    import json
    (OUT / "profile.json").write_text(json.dumps(prof, indent=2, default=str))
    print(f"\nfull profile -> {OUT/'profile.json'}", flush=True)
    print("WAVLM_PROFILE_RESULTS_DONE", flush=True)


if __name__ == "__main__":
    main()
