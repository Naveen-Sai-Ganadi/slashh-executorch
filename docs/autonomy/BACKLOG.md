# Slashh AI — Backlog

On-device voice-stress detection: PyTorch → ExecuTorch `.pte` → Snapdragon NPU,
fully offline (airplane-mode), with a calming intervention. Source of truth for
the autonomy loop. Items are ordered; `done` kept at the bottom for history.
See `docs/plan.md` for the full design and section references.

---

## [M5] AI Hub INT8/QNN/NPU profiling — the 40% evidence
- **status:** todo
- **priority:** P0
- **role:** export-engineer
- **depends-on:** [M2]
- **acceptance:**
  - `model/aihub_profile.py --submit` compiles StressNet for "Samsung Galaxy S25 Ultra" (os 15) and returns a QNN target model
  - INT8 quantize job succeeds with the synthetic/real calibration set
  - profile job reports NPU latency + throughput; numbers recorded in `docs/benchmarks/`
  - on-device inference parity vs eager within agreed tolerance (INT8 expected looser than fp32 1e-4)
  - **gate:** requires explicit human go-ahead — submitting uses the live token and consumes credits

## [M7] On-device ExecuTorch inference
- **status:** in-progress
- **priority:** P0
- **role:** runtime-engineer
- **depends-on:** [M2, M6]
- **acceptance:**
  - ✅ `ExecuTorchStressClassifier` wraps `Module.load` + `forward` on the `[1,1,64,301]` tensor (written; behind the `StressClassifier` interface)
  - ✅ `MainActivity` copies the `.pte` out of assets and runs the live mic→score loop
  - ⬜ verify on a device: ExecuTorch AAR resolves, runtime loads the `.pte`, no leak over a long session
  - ⬜ on-device score matches host `run_pte` on the same feature tensor within tolerance
  - **note:** needs a JDK17 + Android Studio / device; not buildable on the current host (no JDK)

## [M8] Stress meter UI
- **status:** done (night build — branch `feature/night-vad-fix-and-jvm-tests`)
- **priority:** P1
- **role:** runtime-engineer
- **depends-on:** [M7]
- **acceptance:**
  - live meter driven by the EMA-smoothed score (`EMA_ALPHA`), crosses at `STRESS_THRESHOLD` / releases at `RELEASE_THRESHOLD` ✅
  - no flicker at the boundary (hysteresis observed) ✅
- **shipped:**
  - `ui/Meter.kt` — pure-JVM view-model: `StressState → MeterModel {percent, band(IDLE/CALM/ELEVATED/HIGH), label, argb, stressed}`. Band agrees with the pipeline latch (`state.stressed` ⇒ HIGH even inside the hysteresis band), so the meter never contradicts the decision loop.
  - `ui/StressMeterView.kt` — thin custom `View` rendering `MeterModel` (percentage + band-colored proportional bar + label); no logic.
  - `MainActivity` rewired from a `TextView` to `StressMeterView`.
  - `MeterTest.kt` — 5 JVM unit tests (idle, calm/green, elevated/amber, latched-HIGH-in-band, percent round/clamp). `run_jvm_tests.sh` generalized to discover test classes across packages → **14 JVM tests green** (was 9).

## [M9] Calming intervention
- **status:** todo
- **priority:** P1
- **role:** runtime-engineer
- **depends-on:** [M8]
- **acceptance:**
  - sustained stress triggers a calming cue (breathing prompt / haptic)
  - intervention is dismissible and rate-limited (no nagging)

## [M10] Airplane-mode hardening
- **status:** todo
- **priority:** P0
- **role:** qa-engineer
- **depends-on:** [M7]
- **acceptance:**
  - full mic→score→intervention loop works with networking disabled
  - no outbound network calls at runtime (verified via network inspection / no INTERNET dependence in the hot path)

## [M11] Benchmarks + docs
- **status:** todo
- **priority:** P1
- **role:** devrel-lead
- **depends-on:** [M5, M7]
- **acceptance:**
  - `docs/benchmarks/` captures XNNPACK-CPU vs QNN-NPU latency, model size, accuracy
  - README quickstart reproduces train → export → run on a clean checkout

---

## [M1] PyTorch stress classifier — DONE
- **status:** done
- **priority:** P0
- **role:** export-engineer
- **depends-on:** none
- **acceptance:**
  - ✅ `StressNet` (3× conv block → GAP → linear → sigmoid) emits a single score in [0,1]
  - ✅ `model/features.py` log-mel extractor + `audio_config` single source of truth
  - ✅ `model/train.py` smoke-trains (synthetic) and saves `{"model": state_dict}`; real data via `--data-dir`
  - ✅ `model/eval.py` reports accuracy/precision/recall + optional whole-set `.pte` parity

## [M2] ExecuTorch export (XNNPACK) — DONE
- **status:** done
- **priority:** P0
- **role:** export-engineer
- **depends-on:** [M1]
- **acceptance:**
  - ✅ `model/export_executorch.py` lowers via `to_edge_transform_and_lower(XnnpackPartitioner)` to a `.pte` (~100 KB)
  - ✅ bundled `flatc` auto-located (`FLATC_EXECUTABLE`) so export works off-PATH / under hooks
  - ✅ `model/run_pte.py` executes `forward` through the ExecuTorch runtime
  - ✅ `tests/test_parity.py`: exported `.pte` matches eager within **1e-4** max abs error
  - ✅ `tests/test_features.py`: feature shape/determinism/finiteness contract pinned

## [M6] Android audio pipeline — mic → VAD → log-mel — DONE
- **status:** done
- **priority:** P0
- **role:** runtime-engineer
- **depends-on:** none
- **acceptance:**
  - ✅ 16 kHz mono capture, 3 s window / 1 s hop (`AudioCapture.kt`, matches `audio_config`)
  - ✅ native log-mel extractor (`LogMel.kt` + exact 400-pt `RealDft.kt`) reproduces `model/features.py` within 1e-3 max abs error on shared golden vectors (`LogMelParityTest`)
  - ✅ VAD (`Vad.kt`, energy floor + ZCR band) gates silence so idle audio doesn't drive the meter
  - ✅ pipeline glue (`StressPipeline.kt`): VAD gate → features → score → EMA → hysteresis, with JVM unit tests (`StressPipelineTest`, `VadTest`)
  - **note:** parity verified algorithmically (literal Python port of `LogMel.kt` vs the golden JSON, ≤1.2e-5); the JUnit `LogMelParityTest` runs in Android Studio/CI — no JDK on the current host
