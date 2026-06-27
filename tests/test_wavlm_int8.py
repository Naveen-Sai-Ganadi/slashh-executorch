"""Surgical INT8 (w8a8) quantization of the WavLM teacher (TDD, red first).

Backlog #17: shrink the LoRA-merged WavLM teacher (~317M params) to an INT8
ExecuTorch ``.pte`` (~317 MB, w8a8) for the host/XNNPACK runtime. A *global*
PT2E pass crashes during calibration — observers land on WavLM's gated
relative-position bias path (``gru_rel_pos_linear`` -> bias -> SDPA), where an
int scalar used as a Long index is rewritten to a float tensor attr by
``transform_for_annotation``. The surgical fix:

  1. ``SafeQuantizer`` overrides ``transform_for_annotation`` to a no-op (we only
     quantize Linears, whose correctness never needs the scalar->attr rewrite).
  2. A node filter EXCLUDES the rel-pos modules so observers never touch the
     Long-index path; everything else (q/k/v/out_proj + the param-heavy FFN
     linears) quantizes normally.
  3. Per-tensor (not per-channel) config: WavLM's hand-rolled attention matmuls
     aren't XNNPACK-partitioned, so their weight-dequant stays in the graph, and
     ``dequantize_per_channel`` has no portable out-variant (serialization
     fails) while ``dequantize_per_tensor`` does.

The pure filter/quantizer logic is unit-tested with hand-built fx nodes; the
quantize plumbing is exercised end-to-end on a tiny MLP (no 317M download), and
the heavy WavLM build/parity is a documented opt-in step, not a unit test.

Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

import types

import torch
from torch import nn


def _fake_node(stack):
    """A torch.fx-node stand-in carrying just the ``nn_module_stack`` meta the
    path extractor reads. ``stack`` is the dict fx puts there: {key: (path, type)}."""
    return types.SimpleNamespace(meta={} if stack is None else {"nn_module_stack": stack})


# --- pure: module-path extraction ------------------------------------------

def test_node_module_paths_extracts_tuple_stack():
    from model.wavlm_int8 import node_module_paths

    node = _fake_node({
        "k0": ("encoder.layers.0.attention.q_proj", object),
        "k1": ("encoder.layers.0", object),
    })
    paths = node_module_paths(node)
    assert "encoder.layers.0.attention.q_proj" in paths
    assert "encoder.layers.0" in paths


def test_node_module_paths_empty_when_no_stack():
    from model.wavlm_int8 import node_module_paths

    assert node_module_paths(_fake_node(None)) == []


def test_node_module_paths_tolerates_non_tuple_values():
    from model.wavlm_int8 import node_module_paths

    # Some stacks store a bare string rather than (path, type); don't crash.
    paths = node_module_paths(_fake_node({"k": "encoder.head"}))
    assert "encoder.head" in paths


# --- pure: the exclude filter ----------------------------------------------

def test_exclude_filter_rejects_excluded_module():
    from model.wavlm_int8 import make_exclude_filter

    keep = make_exclude_filter(("gru_rel_pos_linear",))
    relpos = _fake_node({"k": ("encoder.layers.3.attention.gru_rel_pos_linear", object)})
    assert keep(relpos) is False


def test_exclude_filter_keeps_unmatched_module():
    from model.wavlm_int8 import make_exclude_filter

    keep = make_exclude_filter(("gru_rel_pos_linear",))
    ffn = _fake_node({"k": ("encoder.layers.3.feed_forward.intermediate_dense", object)})
    assert keep(ffn) is True


def test_exclude_filter_keeps_nodes_without_stack():
    from model.wavlm_int8 import make_exclude_filter

    # No module stack (e.g. a free op) must be kept, not silently dropped.
    keep = make_exclude_filter(("anything",))
    assert keep(_fake_node(None)) is True


def test_relpos_constant_covers_the_crashing_path():
    from model.wavlm_int8 import RELPOS_MODULES, make_exclude_filter

    # The documented crash is on gru_rel_pos_linear; the default exclude set
    # must reject exactly that node.
    keep = make_exclude_filter(RELPOS_MODULES)
    crash = _fake_node({"k": ("encoder.layers.0.attention.gru_rel_pos_linear", object)})
    assert keep(crash) is False


# --- SafeQuantizer: the transform_for_annotation no-op ----------------------

def test_safe_quantizer_transform_is_identity():
    from model.wavlm_int8 import SafeQuantizer

    q = SafeQuantizer()
    m = nn.Linear(4, 4)
    # Must return the model unchanged (the crashing scalar->attr rewrite skipped).
    assert q.transform_for_annotation(m) is m


# --- end-to-end plumbing on a tiny MLP (no 317M model) ----------------------

def test_quantize_int8_tiny_mlp_runs_and_agrees(tmp_path):
    from model.run_pte import run_pte
    from model.wavlm_int8 import quantize_int8

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)).eval()
    calib = torch.randn(16, 64)

    pte = quantize_int8(model, calib, exclude=())
    assert isinstance(pte, (bytes, bytearray)) and len(pte) > 0

    path = tmp_path / "toy_int8.pte"
    path.write_bytes(pte)

    x = torch.randn(8, 64)
    agree, dmax = 0, 0.0
    with torch.no_grad():
        for i in range(x.shape[0]):
            r = model(x[i:i + 1]).flatten()
            q = run_pte(str(path), x[i:i + 1]).flatten()
            dmax = max(dmax, (r - q).abs().max().item())
            agree += int((r > 0).item() == (q > 0).item())
    # INT8 of a tiny linear net should track fp32 closely and mostly agree.
    assert agree >= 6
    assert dmax < 1.0


# --- parity reduction logic (no model) -------------------------------------

def test_parity_metrics_shape_and_agreement():
    from model.wavlm_int8 import parity_metrics

    fp32 = torch.tensor([-2.5, 1.2, -0.3, 3.0])
    int8 = torch.tensor([-2.1, 0.9, 0.1, 2.4])  # 3rd flips sign (-0.3 -> +0.1)
    # |Δ| = [0.4, 0.3, 0.4, 0.6] -> max 0.6, mean 0.425
    m = parity_metrics(fp32, int8)
    assert m["n"] == 4
    assert m["decision_agree"] == 3
    assert m["decision_agree_rate"] == 0.75
    assert abs(m["max_abs_delta"] - 0.6) < 1e-5
    assert abs(m["mean_abs_delta"] - 0.425) < 1e-5


def test_parity_metrics_with_labels_reports_accuracy():
    from model.wavlm_int8 import parity_metrics

    fp32 = torch.tensor([-2.0, 2.0, -1.0, 1.0])   # preds: 0,1,0,1
    int8 = torch.tensor([-1.5, 1.5, 0.5, 0.8])    # preds: 0,1,1,1
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    m = parity_metrics(fp32, int8, labels=labels)
    assert m["fp32_accuracy"] == 1.0      # all 4 correct
    assert m["int8_accuracy"] == 0.75     # 3rd wrong
