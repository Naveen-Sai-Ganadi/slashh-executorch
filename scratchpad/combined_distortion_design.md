# Design brief — combined-distortion-robustness

First **simultaneous** multi-distortion sweep. All prior robustness items isolate ONE
axis; this composes them in physical signal-chain order and quantifies the
**compounding gap** vs single-axis predictions. Mirrors the shipped siblings
`model/reverb_robustness.py`, `model/clipping_robustness.py`, `model/gain_robustness.py`
exactly in shape (pure reduction unit-tested with hand-set numbers; `build_*` trains one
tiny net and evaluates distorted eval sets).

## Module: `model/combined_distortion.py`

### Physical signal-chain order (FIXED): gain → reverb → noise → clip
Rationale: input level (gain) happens at the mic, the room convolves it (reverb),
ambient noise adds at the mic sum (noise), then the ADC/preamp hard-clips (clip). This
is the realistic order and must be applied in this sequence.

### Public API (`__all__`)
```python
@dataclass(frozen=True)
class FieldProfile:
    name: str
    gain_db: float          # input level; factor = 10**(gain_db/20), applied FIRST
    rt60_s: float           # reverb time s; 0.0 = dry; applied SECOND (convolutive)
    snr_db: float | None    # additive noise SNR dB; None = no noise; applied THIRD
    clip_ratio: float       # hard-clip thr ratio; 1.0 = no clip; applied LAST
    noise_color: str = "white"
    def is_clean(self) -> bool   # gain_db==0 and rt60_s==0 and snr_db is None and clip_ratio>=1.0

# Named field profiles (severities drawn from the already-shipped single-axis sweeps
# so they are directly matchable). Pick concrete values in these ranges:
#   clean      : 0dB,   0.0s,  None, 1.0          (degenerate identity)
#   quiet      : -3dB,  0.15s, 20.0, 0.9          (small quiet office)
#   typical    : -6dB,  0.30s, 10.0, 0.6          (normal room + background)
#   harsh      : -12dB, 0.60s, 0.0,  0.35         (loud reverberant room, clipping mic)
#   worst_case : -12dB, 1.0s,  -5.0, 0.25         (corner: ~worst of each shipped axis)
DEFAULT_PROFILES: tuple[FieldProfile, ...]

def apply_field_profile(wave, profile: FieldProfile, gen) -> tuple["Tensor", dict]:
    """Compose gain→reverb→noise→clip. Returns (distorted_wave, diagnostics).
    diagnostics = {"tail_fraction":..., "clipped_fraction":...}.
    A clean/identity profile MUST return the wave bit-for-bit unchanged (no RNG draw),
    so a degenerate profile reproduces clean features exactly.
    Reuse the proven injectors: import _add_noise from .robustness, replicate the
    _reverb_wave exp-decay RIR and _clip_wave hard-clip logic (or import if exposed).
    Gain = wave * 10**(gain_db/20)."""

@dataclass(frozen=True)
class ProfilePoint:
    name: str
    gain_db: float; rt60_s: float; snr_db: float | None; clip_ratio: float
    accuracy: float
    acc_std: float | None      # std across eval seeds (None if single-seed)
    n: int
    matched_single_axis_acc: float | None = None  # min single-axis acc at matched severities
    def reliable(self, bar: float) -> bool   # accuracy >= bar
    def compounding_gap(self) -> float | None # matched_single_axis_acc - accuracy (>=0 = compounding hurts)
    def to_dict(self) -> dict

@dataclass(frozen=True)
class CombinedDistortionResult:
    points: tuple[ProfilePoint, ...]
    accuracy_bar: float
    # helpers:
    def clean_accuracy(self) -> float | None        # the is_clean point's accuracy
    def worst_accuracy(self) -> float
    def reliable_profiles(self) -> tuple[str, ...]   # names with accuracy>=bar
    def max_compounding_gap(self) -> float | None    # largest gap across non-clean profiles
    def graceful(self) -> bool                       # all profiles reliable
    def verdict(self) -> str                         # honest one-liner (see below)
    def to_dict(self) -> dict

def combined_distortion(records, *, single_axis=None, accuracy_bar=0.8,
                        out_dir="docs/benchmarks") -> CombinedDistortionResult:
    """PURE reduction. `records` = list of tuples
    (name, gain_db, rt60_s, snr_db, clip_ratio, accuracy, acc_std, n).
    `single_axis` (optional) = dict name -> matched_single_axis_acc. Writes
    combined_distortion.json + .md when out_dir is not None. No torch import on this path."""

def to_markdown(out: CombinedDistortionResult) -> str: ...

def build_combined_distortion(*, seed=0, epochs=12, n_per_class=96, eval_n_per_class=96,
                              profiles=DEFAULT_PROFILES, eval_seeds=(0,1,2),
                              accuracy_bar=0.8, out_dir="docs/benchmarks"
                              ) -> CombinedDistortionResult:
    """Train ONE production model via model.production.train_production(epochs=, n_per_class=,
    seed=, snr_levels default, threshold=0.8). For each profile: build a balanced eval set
    per eval_seed, apply_field_profile to each waveform, re-extract log-mel via
    model.features.extract, score (pred = model(x)>=0.5), accuracy = mean(pred==y);
    aggregate mean/std across eval_seeds. ALSO compute matched single-axis accuracy =
    min over the profile's ACTIVE axes of {noise-only@snr, reverb-only@rt60, clip-only@ratio,
    gain-only@gain_db} accuracy using the SAME model and SAME eval seeds, so the compounding
    gap is data-backed not asserted. Build records + single_axis dict, call combined_distortion()."""
```

### Verdict logic (honest)
- if `graceful()` (every profile incl. worst_case ≥ bar): "graceful composition — robustness holds under simultaneous distortion".
- elif clean ≥ bar but harsher profiles drop below bar: "compounding degradation — single-axis robustness OVERSTATES field robustness; reliable only through <last reliable profile>; max compounding gap <g>. Fix: combined-distortion augmentation (augmentation_ab / recipe_envelope)."
- else: report honestly.

## Tests: `tests/test_combined_distortion.py` (target ~12–16, MUST be red first)
Keep deterministic — follow the sibling convention (pure reduction tested with hand-set
numbers; ONE cheap integration test). Required:
1. **Injector identity**: `apply_field_profile(wave, CLEAN, gen)` returns wave unchanged
   (torch.equal / allclose 0 atol), and draws no RNG (gen state unchanged) — this is
   acceptance crit (3a) at the injector level.
2. **Injector monotone magnitude**: harsher profile perturbs the wave more than a milder
   one (e.g. L2 distance from clean: harsh > typical > quiet > clean==0).
3. **Order correctness**: clipping is applied last (output respects the clip bound after
   noise added), noise present when snr_db set, reverb changes length-preserved signal.
4. **Pure reduction**: hand-set records → ProfilePoint.accuracy/compounding_gap, Result
   clean_accuracy/worst_accuracy/reliable_profiles/max_compounding_gap/graceful/verdict.
5. **Monotonicity in reduction** (crit 3b): given hand-set monotone records
   (quiet≥typical≥harsh), the result preserves/reports that ordering; given a non-monotone
   set it still reduces without crashing.
6. **Compounding non-negative** (crit 3c): given hand-set records where each profile's
   accuracy ≤ its matched_single_axis_acc, every reported compounding_gap ≥ 0; verdict
   names compounding when a profile falls below bar.
7. **to_dict round-trips / byte-stable keys**; **to_markdown** contains each profile name
   and the verdict.
8. **Index guard** (in test_benchmarks_index or here): the summarizer handles
   combined_distortion.json.
9. **ONE integration test** (cheap: epochs=2, n_per_class=16, eval_n_per_class=16,
   eval_seeds=(0,)): `build_combined_distortion(out_dir=tmp)` runs, writes json+md, the
   clean profile's accuracy equals a separately-computed clean baseline EXACTLY (crit 3a
   end-to-end), and harsh.accuracy ≤ clean.accuracy + small tol (soft; mark as soft to
   avoid flakiness — do NOT assert a hard strict inequality on the tiny stochastic model).

NOTE on crit 3c as a model assertion: encode it as a property of the REDUCTION (hand-set
numbers), not a hard assertion on the trained tiny model (which can be noisy). The
integration test uses soft tolerances.

## Artifacts
- `docs/benchmarks/combined_distortion.json` + `.md`
- wire into `model/benchmarks_index.py` (registry line + `_summarize_combined_distortion`)
- regen `docs/benchmarks/README.md` index section
- one honest line each in README quickstart/benchmarks and CHANGELOG

## Constraints
Host-only. No android/, Vad.kt, run_jvm_tests.sh. No live AI Hub token import on the test
path. Seeded torch.Generators; byte-stable JSON (sorted keys, rounded floats like siblings).
