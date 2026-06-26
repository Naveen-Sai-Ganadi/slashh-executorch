"""fp32 vs INT8 .pte on-disk footprint A/B.

Every other INT8 artifact in this repo measures *accuracy/calibration* under
quantization. None measures what quantization is actually for on-device: the
program shrinks. This A/B exports the SAME trained production model two ways —
fp32 XNNPACK and INT8 PT2E — and compares the serialized .pte byte size, the
concrete cost behind the "INT8 on the NPU" headline.

Honest angle: the production net is 1,549 params, so the .pte's fixed runtime
and flatbuffer/header overhead is a large fraction of the file. Only the weight
constants compress fp32->int8 (a 4x ceiling); fixed overhead does not. The
realized whole-file compression therefore falls short of the textbook 4x — this
module reports the realized ratio *and* how far it lands below the ceiling, so
the number is honest about a tiny model rather than quoting the headline 4x.

Host-only; no device, no AI Hub token. The reduction is pure (unit-tested
without exporting); ``build_pte_footprint`` trains + exports one tiny net.

Run:
    PYTHONPATH=. .venv/bin/python -m model.pte_footprint
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "FootprintResult",
    "pte_footprint",
    "build_pte_footprint",
]

# fp32 (32-bit) -> int8 (8-bit) weights: a 4x ceiling on *weight-constant* bytes.
# Fixed program overhead doesn't compress, so whole-file ratio stays under this.
_THEORETICAL_RATIO = 4.0

# A quantized program "materially" shrinks if it sheds at least this fraction of
# the fp32 file. Below it, the size win doesn't justify the INT8 accuracy risk.
_MATERIAL_SAVINGS_PCT = 25.0


@dataclass(frozen=True)
class FootprintResult:
    """fp32 vs INT8 .pte byte sizes for one model."""

    fp32_bytes: int
    int8_bytes: int
    n_params: int

    @property
    def compression_ratio(self) -> float:
        """fp32 file size / INT8 file size (>1 means INT8 is smaller)."""
        return self.fp32_bytes / self.int8_bytes

    @property
    def savings_pct(self) -> float:
        """Percent of the fp32 file shed by quantizing (negative if it grew)."""
        return (1.0 - self.int8_bytes / self.fp32_bytes) * 100.0

    @property
    def pct_of_theoretical(self) -> float:
        """Realized compression as a percent of the 4x weight-only ceiling."""
        return self.compression_ratio / _THEORETICAL_RATIO * 100.0

    @property
    def fp32_bytes_per_param(self) -> float:
        return self.fp32_bytes / self.n_params

    @property
    def int8_bytes_per_param(self) -> float:
        return self.int8_bytes / self.n_params

    @property
    def material_shrink(self) -> bool:
        return self.savings_pct >= _MATERIAL_SAVINGS_PCT

    @property
    def verdict(self) -> str:
        if self.savings_pct < 0:
            return (
                f"INT8 does *not* shrink this {self.n_params}-param model: the "
                f".pte grows {-self.savings_pct:.1f}% ({self.fp32_bytes}->"
                f"{self.int8_bytes} bytes). At this size the fixed runtime/header "
                "overhead plus per-channel quant metadata outweighs the 4x weight "
                "saving — the size case for quantization is the on-device NPU "
                "(speed/power), not the file."
            )
        ratio_note = (
            f"{self.compression_ratio:.2f}x smaller ({self.savings_pct:.1f}% off, "
            f"{self.fp32_bytes}->{self.int8_bytes} bytes)"
        )
        ceiling_note = (
            f"that is {self.pct_of_theoretical:.0f}% of the 4x weight-only "
            f"ceiling — the gap is fixed .pte overhead that doesn't compress on a "
            f"{self.n_params}-param net"
        )
        if self.material_shrink:
            return (
                f"INT8 materially shrinks the program: {ratio_note}; {ceiling_note}. "
                "The realized file saving is real but quote it, not the textbook 4x."
            )
        return (
            f"INT8 shrinks the program only modestly: {ratio_note}; {ceiling_note}. "
            f"Below the {_MATERIAL_SAVINGS_PCT:g}% bar — the deployment win for this "
            "tiny model is NPU speed/power, not file size."
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compression_ratio"] = self.compression_ratio
        d["savings_pct"] = self.savings_pct
        d["pct_of_theoretical"] = self.pct_of_theoretical
        d["fp32_bytes_per_param"] = self.fp32_bytes_per_param
        d["int8_bytes_per_param"] = self.int8_bytes_per_param
        d["material_shrink"] = self.material_shrink
        d["theoretical_ratio"] = _THEORETICAL_RATIO
        d["verdict"] = self.verdict
        return d


def pte_footprint(
    *,
    fp32_bytes: int,
    int8_bytes: int,
    n_params: int,
    out_dir: str | Path | None = None,
) -> FootprintResult:
    """Compare fp32 vs INT8 .pte byte sizes. Pure — no export, no device.

    All three inputs must be positive. INT8 *not* being smaller is a valid,
    reported outcome (fixed overhead on a tiny model), not an error. Writes
    ``pte_footprint.{json,md}`` to ``out_dir`` when given.
    """
    if fp32_bytes <= 0 or int8_bytes <= 0:
        raise ValueError("fp32_bytes and int8_bytes must be positive")
    if n_params <= 0:
        raise ValueError("n_params must be positive")

    result = FootprintResult(
        fp32_bytes=int(fp32_bytes),
        int8_bytes=int(int8_bytes),
        n_params=int(n_params),
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "pte_footprint.json").write_text(
            json.dumps(result.to_dict(), indent=2) + "\n"
        )
        (out_dir / "pte_footprint.md").write_text(to_markdown(result))

    return result


def to_markdown(out: FootprintResult) -> str:
    d = out.to_dict()
    return (
        "# fp32 vs INT8 .pte footprint (A/B)\n\n"
        "The same trained production net, exported two ways. Quantization's "
        "on-device payoff is a smaller, faster program; this measures the *size* "
        "half honestly — the realized whole-file ratio against the 4x weight-only "
        "ceiling, on a net small enough that fixed overhead matters.\n\n"
        f"- model: **{out.n_params}** params\n"
        f"- fp32 `.pte`: **{out.fp32_bytes:,}** bytes  ·  INT8 `.pte`: "
        f"**{out.int8_bytes:,}** bytes\n"
        f"- compression: **{out.compression_ratio:.2f}x** "
        f"(**{out.savings_pct:+.1f}%**), **{out.pct_of_theoretical:.0f}%** of the "
        f"{_THEORETICAL_RATIO:g}x weight-only ceiling\n"
        f"- materially smaller (>={_MATERIAL_SAVINGS_PCT:g}%): "
        f"**{d['material_shrink']}**\n"
        f"- **Verdict: {out.verdict}**\n\n"
        "| variant | .pte bytes | bytes/param |\n"
        "|---|---|---|\n"
        f"| fp32 (XNNPACK) | {out.fp32_bytes:,} | {out.fp32_bytes_per_param:.1f} |\n"
        f"| INT8 (PT2E) | {out.int8_bytes:,} | {out.int8_bytes_per_param:.1f} |\n"
    )


def build_pte_footprint(
    *,
    seed: int = 0,
    epochs: int = 12,
    n_per_class: int = 96,
    calib_n: int = 24,
    out_dir: str | Path | None = "docs/benchmarks",
) -> FootprintResult:
    """Train one production net, export fp32 + INT8, and compare .pte sizes.

    Imports inside the function to keep the export/quant stack off the pure
    reduction import path.
    """
    from .export_executorch import export_to_pte
    from .production import quantize_production, train_production

    model, _ = train_production(epochs=epochs, n_per_class=n_per_class, seed=seed)
    model.eval()

    fp32 = export_to_pte(model=model, delegate=True)
    int8 = quantize_production(model, calib_n=calib_n, seed=seed + 2024)
    n_params = sum(p.numel() for p in model.parameters())

    return pte_footprint(
        fp32_bytes=len(fp32),
        int8_bytes=len(int8),
        n_params=n_params,
        out_dir=out_dir,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="fp32 vs INT8 .pte footprint A/B")
    p.add_argument("--out-dir", default="docs/benchmarks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--n-per-class", type=int, default=96)
    args = p.parse_args()
    out = build_pte_footprint(
        seed=args.seed, epochs=args.epochs, n_per_class=args.n_per_class,
        out_dir=args.out_dir,
    )
    print(to_markdown(out))
    print(f"wrote {args.out_dir}/pte_footprint.json and .md")


if __name__ == "__main__":
    main()
