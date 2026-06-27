"""Extract the raw QNN context binary embedded in an ExecuTorch QNN .pte.

ExecuTorch wraps the QNN context binary in a custom protocol (QnnContextCustomProtocol).
`dump_context_from_pte` deserializes the .pte, pulls each QnnBackend delegate's
processed_bytes and calls PyQnnManagerAdaptor.StripProtocol(...) to recover the raw
QNN context binary (`<method>_<idx>.bin`). That .bin is directly runnable on the
Hexagon HTP with the prebuilt qnn-net-run --retrieve_context.

Runs inside the linux/amd64 container (ET native QNN adaptor is x86-linux only).
"""
import sys
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte

pte = sys.argv[1] if len(sys.argv) > 1 else "assets/teacher_wavlm_broad_qnn.pte"
print(f"[extract] reading {pte}", flush=True)
files = dump_context_from_pte(pte)
print(f"[extract] dumped {len(files)} context binary file(s):", flush=True)
import os
for f in files:
    print(f"  {f}  ({os.path.getsize(f)} bytes)", flush=True)
if not files:
    print("[extract] ERROR: no QnnBackend delegate found in .pte", flush=True)
    sys.exit(2)
