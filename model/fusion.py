"""Late-fusion model: [audio_score, text_score] -> fused stress in [0,1].

Run as a module:

    python -m model.fusion

What it does (idempotent / re-runnable):
  1. TEXT score distribution per label — run the trained text model
     (:mod:`model.text_stress`) on ALL Dreaddit rows (train+test) and collect the
     model's stress scores split by the true label (y=0 / y=1).
  2. AUDIO score distribution per label — load the existing trained StressNet
     audio model (``assets/stress_model.pt``) and score RAVDESS windows
     (angry+fearful => stressed, neutral+calm => calm). On this repo's checkpoint
     the audio model was trained on *synthetic* audio and its RAVDESS scores
     collapse to ~0 for BOTH classes (no usable [0,1] spread), so we FALL BACK to
     Beta-distributed audio scores (calm mean ~0.30, stressed mean ~0.70). The
     fallback is reported loudly. See :func:`audio_score_distributions`.
  3. Build ~20000 paired samples: y~Bernoulli(0.5); audio ~ audio_scores[y];
     text ~ text_scores[y]. Train a tiny MLP — ``Linear(2,16) -> ReLU ->
     Linear(16,1)`` with ``sigmoid`` in forward — on (audio,text)->y (BCE).
  4. Save ``assets/fusion.pt`` and export ``android/app/src/main/assets/fusion.pte``
     (same XNNPACK fp32 pattern). Print the 5x5 fusion grid for verification.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

from .audio_config import SAMPLE_RATE, WINDOW_SAMPLES
from .export_executorch import _ensure_flatc
from .text_features import load_vocab, text_to_vector
from .text_stress import TRAIN_CSV, TEST_CSV, load_text_model

# --- Paths / config (the contract) ---
AUDIO_CKPT = Path("assets/stress_model.pt")
CHECKPOINT_PATH = Path("assets/fusion.pt")
PTE_PATH = Path("android/app/src/main/assets/fusion.pte")
RAVDESS_GLOB = "data/Actor_*"

N_PAIRS = 20000
SEED = 0

# Decision-boundary calibration: after training, shift the perceptron bias so fused crosses 0.5
# only when audio + text >= this sum (i.e. BOTH signals ~0.6+). Higher => more conservative
# (fewer false "stressed" calls on ordinary speech). 1.2 = the tuned "moderate" setting.
DECISION_BOUNDARY_SUM = 1.2

# RAVDESS emotion code (3rd field of filename) -> stress label.
# 05 angry + 06 fearful => stressed(1); 01 neutral + 02 calm => calm(0).
EMOTION_TO_LABEL = {"05": 1, "06": 1, "01": 0, "02": 0}

# Beta fallback params: mean = a/(a+b); a+b=8 gives moderate spread (std ~0.15).
# calm  mean ~0.30 -> Beta(2.4, 5.6); stressed mean ~0.70 -> Beta(5.6, 2.4).
BETA_CALM = (2.4, 5.6)
BETA_STRESSED = (5.6, 2.4)
# If the real audio model's per-label means differ by less than this, treat the
# distributions as degenerate (no usable signal) and fall back to Beta.
MIN_MEAN_SEPARATION = 0.10


class FusionNet(nn.Module):
    """Late-fusion **single perceptron** (logistic regression): Linear(2,1) -> sigmoid.

    Input [B,2] float32 = [audio_score, text_score] in [0,1]; output [B,1] in [0,1].
    Deliberately a plain linear model (not an MLP) so the fusion is smooth, monotonic,
    and — with swap-augmented training (see :func:`build_pairs`) — **symmetric** in its
    two inputs: audio and text contribute equally, neither dominates. The decision is
    essentially "stress when audio + text are jointly elevated", which is what a human
    expects when watching the two signals. Only Linear/Sigmoid → clean XNNPACK lowering.
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def example_input(batch: int = 1) -> torch.Tensor:
    """Representative input for export / smoke tests: [batch, 2] float32."""
    return torch.randn(batch, 2, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Score distributions per label                                               #
# --------------------------------------------------------------------------- #
def text_score_distributions() -> dict[int, np.ndarray]:
    """Run the trained text model on ALL Dreaddit rows; split scores by label.

    Returns ``{0: scores_calm, 1: scores_stressed}`` (numpy float32 arrays).
    """
    model = load_text_model()
    vocab = load_vocab()

    frames = [pd.read_csv(TRAIN_CSV), pd.read_csv(TEST_CSV)]
    df = pd.concat(frames, ignore_index=True)
    texts = df["text"].astype(str).tolist()
    labels = df["label"].values.astype(int)

    rows = [text_to_vector(t, vocab) for t in texts]
    x = torch.cat(rows, dim=0)
    with torch.no_grad():
        scores = model(x).squeeze(1).numpy().astype(np.float32)

    return {
        0: scores[labels == 0],
        1: scores[labels == 1],
    }


def _load_wav_window(path: str) -> torch.Tensor:
    """Load a WAV (soundfile), mono, resample to SAMPLE_RATE, center window."""
    import soundfile as sf
    import torchaudio

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(data).mean(dim=1)
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    if wave.numel() >= WINDOW_SAMPLES:
        start = (wave.numel() - WINDOW_SAMPLES) // 2
        wave = wave[start:start + WINDOW_SAMPLES]
    else:
        wave = torch.nn.functional.pad(wave, (0, WINDOW_SAMPLES - wave.numel()))
    return wave


def _real_audio_scores() -> dict[int, np.ndarray] | None:
    """Score RAVDESS windows with the trained audio StressNet, split by label.

    Returns ``{0: calm_scores, 1: stressed_scores}`` or ``None`` if the audio
    checkpoint / data is unavailable. The caller decides whether the resulting
    distributions are usable (see :func:`audio_score_distributions`).
    """
    if not AUDIO_CKPT.exists():
        return None
    try:
        from .model import build_model
        from .features import extract
    except Exception:
        return None

    model = build_model(str(AUDIO_CKPT))
    model.eval()

    per_label: dict[int, list[float]] = {0: [], 1: []}
    for emotion_code, label in EMOTION_TO_LABEL.items():
        paths = sorted(glob.glob(f"{RAVDESS_GLOB}/03-01-{emotion_code}-*.wav"))
        for path in paths:
            try:
                wave = _load_wav_window(path)
                x = extract(wave)
                with torch.no_grad():
                    per_label[label].append(float(model(x).item()))
            except Exception:
                continue

    if not per_label[0] or not per_label[1]:
        return None
    return {0: np.asarray(per_label[0], dtype=np.float32),
            1: np.asarray(per_label[1], dtype=np.float32)}


def _beta_audio_scores(rng: np.random.Generator, n: int = 4000) -> dict[int, np.ndarray]:
    """Beta-distributed audio score samples (the documented fallback)."""
    calm = rng.beta(*BETA_CALM, size=n).astype(np.float32)
    stressed = rng.beta(*BETA_STRESSED, size=n).astype(np.float32)
    return {0: calm, 1: stressed}


def audio_score_distributions(
    rng: np.random.Generator,
) -> tuple[dict[int, np.ndarray], str, dict]:
    """Per-label audio score distributions + which source was used + diagnostics.

    Tries the real trained audio model on RAVDESS first. If the per-label means
    are not separated by at least ``MIN_MEAN_SEPARATION`` (the repo's audio
    checkpoint was trained on synthetic audio and collapses to ~0 on RAVDESS),
    falls back to Beta distributions and records why.
    """
    diag: dict = {}
    real = _real_audio_scores()
    if real is not None:
        calm_mean = float(real[0].mean())
        stressed_mean = float(real[1].mean())
        diag["real_calm_mean"] = calm_mean
        diag["real_stressed_mean"] = stressed_mean
        diag["real_calm_n"] = int(real[0].size)
        diag["real_stressed_n"] = int(real[1].size)
        separation = stressed_mean - calm_mean
        diag["real_mean_separation"] = separation
        if separation >= MIN_MEAN_SEPARATION:
            return real, "real_audio_model", diag
        diag["fallback_reason"] = (
            f"real audio model RAVDESS scores degenerate: stressed mean "
            f"{stressed_mean:.5f} vs calm mean {calm_mean:.5f} "
            f"(separation {separation:.5f} < {MIN_MEAN_SEPARATION}); the "
            f"checkpoint was trained on synthetic audio and does not span [0,1] "
            f"on RAVDESS, so it cannot provide usable per-label audio scores."
        )
    else:
        diag["fallback_reason"] = (
            "real audio model/RAVDESS unavailable (missing checkpoint, data, or "
            "decode failure)."
        )

    beta = _beta_audio_scores(rng)
    diag["beta_calm_mean"] = float(beta[0].mean())
    diag["beta_stressed_mean"] = float(beta[1].mean())
    return beta, "beta_fallback", diag


# --------------------------------------------------------------------------- #
# Paired training data + training                                             #
# --------------------------------------------------------------------------- #
def build_pairs(
    audio_scores: dict[int, np.ndarray],
    text_scores: dict[int, np.ndarray],
    rng: np.random.Generator,
    n: int = N_PAIRS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw ~n paired samples under conditional independence given the label.

    y ~ Bernoulli(0.5); audio ~ audio_scores[y]; text ~ text_scores[y].
    Returns (X [n,2] float32 = [audio, text], y [n,1] float32).
    """
    y = rng.integers(0, 2, size=n)
    audio = np.empty(n, dtype=np.float32)
    text = np.empty(n, dtype=np.float32)
    for label in (0, 1):
        mask = y == label
        k = int(mask.sum())
        a_pool = audio_scores[label]
        t_pool = text_scores[label]
        audio[mask] = a_pool[rng.integers(0, a_pool.size, size=k)]
        text[mask] = t_pool[rng.integers(0, t_pool.size, size=k)]
    x = torch.from_numpy(np.stack([audio, text], axis=1))
    y_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)
    # Symmetry augmentation: also include each pair SWAPPED — (text, audio) with the
    # same label — so the perceptron treats its two inputs interchangeably and neither
    # modality can dominate. This is what fixes "audio low => fused ~0 even when text is
    # high": now high text contributes just as much as high audio.
    x_swap = torch.from_numpy(np.stack([text, audio], axis=1))
    x = torch.cat([x, x_swap], dim=0)
    y_t = torch.cat([y_t, y_t], dim=0)
    return x, y_t


def train_fusion(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    epochs: int = 200,
    batch_size: int = 256,
    lr: float = 5e-3,
    weight_decay: float = 1e-5,
    seed: int = SEED,
) -> FusionNet:
    """Train the fusion MLP with BCE. Returns the model in eval mode."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = FusionNet()
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(x.shape[0], generator=g)
        for i in range(0, x.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            logits = model.net(x[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
    model.eval()
    return model


def fusion_grid(
    model: FusionNet, levels=(0.0, 0.25, 0.5, 0.75, 1.0)
) -> list[tuple[float, float, float]]:
    """Evaluate the fused output over the audio x text grid. (audio, text, score)."""
    grid = []
    model.eval()
    with torch.no_grad():
        for a in levels:
            for t in levels:
                inp = torch.tensor([[a, t]], dtype=torch.float32)
                grid.append((a, t, float(model(inp).item())))
    return grid


def export_to_pte(model: FusionNet) -> bytes:
    """Serialize the fusion model to ``.pte`` bytes (XNNPACK fp32 delegate)."""
    _ensure_flatc()
    model.eval()
    example = (example_input(),)
    exported = torch.export.export(model, example)
    edge = to_edge_transform_and_lower(exported, partitioner=[XnnpackPartitioner()])
    return edge.to_executorch().buffer


def load_fusion_model(checkpoint: str | Path = CHECKPOINT_PATH) -> FusionNet:
    """Rebuild the trained fusion model from its checkpoint (eval mode)."""
    state = torch.load(checkpoint, map_location="cpu")
    model = FusionNet()
    model.load_state_dict(state["model"])
    model.eval()
    return model


def _check_monotonic(grid: list[tuple[float, float, float]]) -> bool:
    """Rising in both inputs: non-decreasing along audio and along text."""
    levels = sorted({a for a, _, _ in grid})
    table = {(a, t): s for a, t, s in grid}
    ok = True
    # Non-decreasing in text for fixed audio.
    for a in levels:
        row = [table[(a, t)] for t in levels]
        ok = ok and all(row[i] <= row[i + 1] + 1e-6 for i in range(len(row) - 1))
    # Non-decreasing in audio for fixed text.
    for t in levels:
        col = [table[(a, t)] for a in levels]
        ok = ok and all(col[i] <= col[i + 1] + 1e-6 for i in range(len(col) - 1))
    return ok


def run(verbose: bool = True) -> dict:
    """Build score distributions, train fusion, save + export, verify grid."""
    rng = np.random.default_rng(SEED)

    text_scores = text_score_distributions()
    audio_scores, audio_source, audio_diag = audio_score_distributions(rng)

    x, y = build_pairs(audio_scores, text_scores, rng)
    model = train_fusion(x, y)

    # Calibrate the decision boundary to be more conservative: keep the learned (symmetric)
    # weights but set the bias so fused = 0.5 exactly when audio + text == DECISION_BOUNDARY_SUM.
    # This requires BOTH signals to be moderately elevated before it calls "stressed", cutting
    # false positives on ordinary speech without reintroducing the old audio-dominant behaviour.
    with torch.no_grad():
        wsum = float(model.net.weight.sum())
        model.net.bias.fill_(-(DECISION_BOUNDARY_SUM / 2.0) * wsum)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, CHECKPOINT_PATH)

    buffer = export_to_pte(model)
    PTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PTE_PATH.write_bytes(buffer)

    grid = fusion_grid(model)
    monotonic = _check_monotonic(grid)
    table = {(a, t): s for a, t, s in grid}
    agree_high = table[(0.75, 0.75)]
    disagree = table[(0.75, 0.25)]

    metrics = {
        "audio_source": audio_source,
        "audio_diag": audio_diag,
        "text_calm_mean": float(text_scores[0].mean()),
        "text_stressed_mean": float(text_scores[1].mean()),
        "audio_calm_mean": float(audio_scores[0].mean()),
        "audio_stressed_mean": float(audio_scores[1].mean()),
        "grid": grid,
        "monotonic": monotonic,
        "pte_bytes": len(buffer),
    }

    if verbose:
        print(f"audio source     : {audio_source}")
        if "fallback_reason" in audio_diag:
            print("AUDIO FALLBACK USED -> " + audio_diag["fallback_reason"])
        if "real_calm_mean" in audio_diag:
            print(
                f"  real audio means  calm={audio_diag['real_calm_mean']:.5f} "
                f"stressed={audio_diag['real_stressed_mean']:.5f} "
                f"(n {audio_diag['real_calm_n']}/{audio_diag['real_stressed_n']})"
            )
        print(
            f"audio score means : calm={metrics['audio_calm_mean']:.4f} "
            f"stressed={metrics['audio_stressed_mean']:.4f}"
        )
        print(
            f"text  score means : calm={metrics['text_calm_mean']:.4f} "
            f"stressed={metrics['text_stressed_mean']:.4f}"
        )
        levels = (0.0, 0.25, 0.5, 0.75, 1.0)
        print("5x5 fusion grid (rows=audio, cols=text):")
        header = "audio\\text " + " ".join(f"{t:>6.2f}" for t in levels)
        print("  " + header)
        for a in levels:
            cells = " ".join(f"{table[(a, t)]:>6.3f}" for t in levels)
            print(f"  {a:>9.2f} {cells}")
        print(f"monotonic (rising in both): {monotonic}")
        print(f"fused(0.75,0.75)={agree_high:.3f}  fused(0.75,0.25)={disagree:.3f}")
        print(f"checkpoint       : {CHECKPOINT_PATH}")
        print(f"pte              : {PTE_PATH} ({len(buffer):,} bytes)")

    return metrics


if __name__ == "__main__":
    run()
