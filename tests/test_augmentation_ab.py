"""Tests for the augmentation-recipe A/B (model/augmentation_ab.py).

robust_train.py proved that *one* noise-augmentation recipe beats clean
training. The detector-noise A/B then showed detector knobs can't push the
10 dB floor deeper — the real lever is training. This harness follows that
through: it sweeps several augmentation recipes (which training-time SNRs to
mix in) over the same architecture and the same eval draw, and recommends the
recipe with the deepest reliable floor that doesn't sacrifice clean accuracy.
Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import json
from pathlib import Path

from model.augmentation_ab import (
    AugRecipe,
    AugmentationABResult,
    augmentation_ab,
)


_RECIPES = [
    AugRecipe(label="clean-only", snr_choices=(None,)),
    AugRecipe(label="mild", snr_choices=(None, 20.0, 10.0)),
]


def test_one_result_per_recipe_with_floor_and_clean_acc() -> None:
    out = augmentation_ab(
        _RECIPES, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=8, seed=0,
    )
    assert isinstance(out, AugmentationABResult)
    assert [r.recipe.label for r in out.results] == ["clean-only", "mild"]
    for r in out.results:
        assert 0.0 <= r.clean_acc <= 1.0
        assert r.reliable_floor_db is None or isinstance(r.reliable_floor_db, float)
        # one accuracy point per eval SNR
        assert [p["snr_db"] for p in r.points] == [None, 10.0, 0.0]


def test_recommends_a_swept_recipe() -> None:
    out = augmentation_ab(
        _RECIPES, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=8, seed=1,
    )
    assert out.recommended is not None
    assert out.recommended.recipe.label in {r.recipe.label for r in out.results}


def test_augmentation_helps_or_holds_floor() -> None:
    # The mild recipe must not have a *shallower* reliable floor than clean-only
    # (augmentation should help or at worst tie on this synthetic task).
    out = augmentation_ab(
        _RECIPES, snr_levels=[None, 10.0, 0.0], n_per_class=64,
        eval_n_per_class=64, epochs=10, seed=0,
    )
    by_label = {r.recipe.label: r for r in out.results}
    clean_floor = by_label["clean-only"].reliable_floor_db
    mild_floor = by_label["mild"].reliable_floor_db
    # deeper (more negative) or holding (None-at-all-SNRs handled by helper);
    # encode "no reliable floor" as +inf so clean-only never looks better.
    def rank(v):
        return float("inf") if v is None else v
    assert rank(mild_floor) <= rank(clean_floor)


def test_deterministic_across_identical_calls() -> None:
    a = augmentation_ab(_RECIPES, snr_levels=[None, 0.0], n_per_class=48,
                        eval_n_per_class=48, epochs=8, seed=2)
    b = augmentation_ab(_RECIPES, snr_levels=[None, 0.0], n_per_class=48,
                        eval_n_per_class=48, epochs=8, seed=2)
    assert a.to_dict() == b.to_dict()


def test_writes_json_and_markdown(tmp_path: Path) -> None:
    augmentation_ab(
        _RECIPES, snr_levels=[None, 10.0, 0.0], n_per_class=48,
        eval_n_per_class=48, epochs=8, seed=0, out_dir=tmp_path,
    )
    assert (tmp_path / "augmentation_ab.json").is_file()
    assert (tmp_path / "augmentation_ab.md").is_file()
    payload = json.loads((tmp_path / "augmentation_ab.json").read_text())
    assert "results" in payload and "recommended" in payload
    md = (tmp_path / "augmentation_ab.md").read_text()
    assert "reliable floor" in md.lower()
