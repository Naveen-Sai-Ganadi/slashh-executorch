"""Slashh AI — on-device voice stress detection (AOT / model side).

PyTorch model + log-mel features + ExecuTorch export. The exported ``.pte`` is
the contract handed to the on-device runtime (Android). Feature extraction
(log-mel) is computed on-device natively; the ``.pte`` is the classifier only,
operating on a fixed-shape log-mel tensor — keep it that way so it quantizes and
lowers cleanly to the NPU.
"""
