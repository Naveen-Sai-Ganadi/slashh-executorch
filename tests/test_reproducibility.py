"""Reproducibility guarantee for the production recipe (model/production.py).

The README sells the model as **reproducible** — "train -> export -> run works
on a clean checkout" — and the benchmark records under docs/benchmarks/ are only
auditable if re-running the recipe reproduces them exactly. This pins that:

  * two ``train_production(seed=0)`` runs produce identical val accuracy and
    bit-identical weights, and
  * the exported ``.pte`` is byte-identical.

Crucially the runs are *not* externally seeded and have unrelated global-RNG use
between them — ``train_production`` seeds itself before building the net, so the
result is independent of call order. That order-independence is the exact
property a prior bug violated (weight init drew from leftover global RNG), so
this test guards against its return. A different seed must still change the
weights, proving the seed is actually wired through. Host-only; no token.
"""

from __future__ import annotations

import torch

from model.export_executorch import export_to_pte
from model.production import train_production


def _flat_weights(model) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def _run():
    model, meta = train_production(epochs=8, n_per_class=64, seed=0)
    return meta["val_acc"], _flat_weights(model), export_to_pte(model=model)


def test_production_recipe_is_bit_reproducible() -> None:
    acc_a, w_a, pte_a = _run()

    # unrelated global-RNG churn between runs — a self-seeding recipe is immune
    torch.randn(1000)
    [torch.rand(()) for _ in range(13)]

    acc_b, w_b, pte_b = _run()

    assert acc_a == acc_b, "val accuracy not reproducible"
    assert torch.equal(w_a, w_b), "weights not bit-identical across runs"
    assert pte_a == pte_b, "exported .pte not byte-identical across runs"


def test_seed_actually_changes_the_model() -> None:
    # sanity that the determinism above isn't because the seed is ignored
    torch.manual_seed(999)
    m0, _ = train_production(epochs=8, n_per_class=64, seed=0)
    m1, _ = train_production(epochs=8, n_per_class=64, seed=1)
    assert not torch.equal(_flat_weights(m0), _flat_weights(m1))
