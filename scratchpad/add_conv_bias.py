"""Inject zero biases into WavLM's bias-less feature-extractor Conv1d ops.

Root cause of the INT8->QNN compile failure (job jp171nm7p):
    preprocessPerChannel: No bias info for op:
    /encoder/feature_extractor/conv_layers.0/conv/Conv_2d
QAIRT's per-channel INT8 weight quantization folds the weight scales into the
conv bias; the wav2vec2/WavLM feature extractor uses bias-less Conv1d (GroupNorm
/LayerNorm follows), so there is no bias tensor and the converter aborts.

Fix: append a zero-valued FLOAT bias initializer (shape [out_channels]) to each
Conv that has only 2 inputs, and wire it as the 3rd input. bias=0 is numerically
inert (the conv output is identical), but it gives QAIRT the per-channel bias it
requires. Writes a new static ONNX; the original is untouched.

    PYTHONPATH="$PWD" .venv/bin/python3 scratchpad/add_conv_bias.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

SRC = Path("assets/teacher_wavlm_broad_static.onnx")
DST = Path("assets/teacher_wavlm_broad_static_bias.onnx")


def main():
    print(f"loading {SRC} ({SRC.stat().st_size/1e9:.2f} GB) …", flush=True)
    m = onnx.load(str(SRC))  # weights embedded; load fully so re-save is complete
    inits = {i.name: i for i in m.graph.initializer}

    patched = []
    for n in m.graph.node:
        if n.op_type != "Conv" or len(n.input) >= 3:
            continue
        w = inits[n.input[1]]
        out_ch = w.dims[0]
        bias_name = f"{n.name}/bias_zero"
        bias = numpy_helper.from_array(
            np.zeros((out_ch,), dtype=np.float32), name=bias_name)
        m.graph.initializer.append(bias)
        n.input.append(bias_name)
        patched.append((n.name, out_ch))

    print(f"patched {len(patched)} bias-less Conv ops:", flush=True)
    for name, ch in patched:
        print(f"   + {bias_name_for(name)}  shape=[{ch}]", flush=True)

    onnx.checker.check_model(m, full_check=False)
    print("onnx.checker OK; saving …", flush=True)
    onnx.save(m, str(DST))
    print(f"wrote {DST} ({DST.stat().st_size/1e9:.2f} GB)", flush=True)

    # verify every Conv now has a bias
    m2 = onnx.load(str(DST), load_external_data=False)
    bad = [x.name for x in m2.graph.node if x.op_type == "Conv" and len(x.input) < 3]
    print("convs still missing bias:", bad if bad else "NONE — all good", flush=True)


def bias_name_for(node_name):
    return f"{node_name}/bias_zero"


if __name__ == "__main__":
    main()
