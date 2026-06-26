"""Tests for the recipe-parameterized cross-init envelope (model/recipe_envelope.py).

The augmentation-recipe A/B (model/augmentation_ab.py) found, on a *single*
seed, that training on lower SNRs deepens the reliable floor (none -> 10 -> 0 ->
-5 dB) at no clean-accuracy cost. But the README's headline number is the
*cross-init envelope* — the floor that holds regardless of the training seed —
which init_envelope.py pins at 10 dB for the production recipe. The open
question that single-seed finding leaves: does a more aggressive recipe move the
**envelope**, or does init variance pull the deeper floor back to 10 dB?

This harness answers that honestly: for each recipe it trains several
independent inits, reduces them to a conservative envelope (worst per-init
floor), and reports the envelopes side by side. Host-only; no device, no AI Hub
token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.recipe_envelope import (
    RecipeEnvelope,
    RecipeEnvelopeResult,
    recipe_envelope,
)
from model.augmentation_ab import AugRecipe


_RECIPES = [
    AugRecipe(label="production", snr_choices=(None, 20.0, 10.0, 5.0)),
    AugRecipe(label="aggressive", snr_choices=(None, 20.0, 10.0, 5.0, 0.0, -5.0)),
]


def test_one_envelope_per_recipe() -> None:
    out = recipe_envelope(
        _RECIPES, n_inits=2, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=6, base_seed=0,
    )
    assert isinstance(out, RecipeEnvelopeResult)
    assert [e.recipe.label for e in out.envelopes] == ["production", "aggressive"]
    for e in out.envelopes:
        assert isinstance(e, RecipeEnvelope)
        # one per-init floor per requested init
        assert len(e.floors) == 2
        assert e.envelope_db is None or isinstance(e.envelope_db, float)
        assert 0.0 <= e.mean_clean_acc <= 1.0


def test_envelope_is_conservative_worst_of_inits() -> None:
    # The envelope must be the *least-noisy* (largest SNR / worst) per-init floor,
    # or None if any init holds only on clean audio.
    out = recipe_envelope(
        _RECIPES, n_inits=2, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=6, base_seed=1,
    )
    for e in out.envelopes:
        floors = [f["reliable_floor_db"] for f in e.floors]
        if any(v is None for v in floors):
            assert e.envelope_db is None
        else:
            assert e.envelope_db == max(floors)


def test_names_deepest_envelope_recipe() -> None:
    out = recipe_envelope(
        _RECIPES, n_inits=2, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=6, base_seed=0,
    )
    # deepest = the recipe whose cross-init envelope reaches the lowest SNR;
    # None envelopes (no common noisy floor) can never be deepest unless all are.
    if out.deepest is not None:
        assert out.deepest.recipe.label in {e.recipe.label for e in out.envelopes}

        def rank(v):
            return float("inf") if v is None else v

        best = min(rank(e.envelope_db) for e in out.envelopes)
        assert rank(out.deepest.envelope_db) == best


def test_deterministic_across_identical_calls() -> None:
    a = recipe_envelope(_RECIPES, n_inits=2, snr_levels=[None, 0.0], n_per_class=48,
                        eval_n_per_class=48, epochs=6, base_seed=2)
    b = recipe_envelope(_RECIPES, n_inits=2, snr_levels=[None, 0.0], n_per_class=48,
                        eval_n_per_class=48, epochs=6, base_seed=2)
    assert a.to_dict() == b.to_dict()


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    recipe_envelope(
        _RECIPES, n_inits=2, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=6, base_seed=0, out_dir=tmp_path,
    )
    assert (tmp_path / "recipe_envelope.json").is_file()
    assert (tmp_path / "recipe_envelope.md").is_file()
    payload = json.loads((tmp_path / "recipe_envelope.json").read_text())
    assert "envelopes" in payload and "deepest" in payload
    md = (tmp_path / "recipe_envelope.md").read_text()
    assert "envelope" in md.lower()
