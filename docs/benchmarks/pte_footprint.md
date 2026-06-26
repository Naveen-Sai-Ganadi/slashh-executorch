# fp32 vs INT8 .pte footprint (A/B)

The same trained production net, exported two ways. Quantization's on-device payoff is a smaller, faster program; this measures the *size* half honestly — the realized whole-file ratio against the 4x weight-only ceiling, on a net small enough that fixed overhead matters.

- model: **1549** params
- fp32 `.pte`: **13,188** bytes  ·  INT8 `.pte`: **12,548** bytes
- compression: **1.05x** (**+4.9%**), **26%** of the 4x weight-only ceiling
- materially smaller (>=25%): **False**
- **Verdict: INT8 shrinks the program only modestly: 1.05x smaller (4.9% off, 13188->12548 bytes); that is 26% of the 4x weight-only ceiling — the gap is fixed .pte overhead that doesn't compress on a 1549-param net. Below the 25% bar — the deployment win for this tiny model is NPU speed/power, not file size.**

| variant | .pte bytes | bytes/param |
|---|---|---|
| fp32 (XNNPACK) | 13,188 | 8.5 |
| INT8 (PT2E) | 12,548 | 8.1 |
