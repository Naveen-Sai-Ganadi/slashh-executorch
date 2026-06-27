# Real-data corpus audit (SP1 recon)

Read-only label characterization of the four emotion-speech corpora downloaded
into `datasets/` on 2026-06-26. **No features extracted, no model code, no
training** — this is reconnaissance to inform the (still-gated) SP1 dataset
design, analogous to the backbone-zoo recon. Source: `scratchpad/audit.py`.

## Arousal mappings compared

- **narrow** = the mapping the *existing* `model/data.py` already uses for
  RAVDESS: `stressed = {angry, fearful}`, `calm = {neutral, calm}`, everything
  else **dropped**.
- **broad** = full Russell circumplex by arousal: `stressed = {angry, fearful,
  happy, surprised, disgust}` (high arousal), `calm = {neutral, calm, sad}`
  (low arousal). Nothing dropped.

## Per-corpus

| Corpus | files | speakers | narrow stressed/calm (used) | broad stressed/calm |
|---|---|---|---|---|
| RAVDESS | 1,440 | 24 | 384 / 288 (672) | 960 / 480 |
| CREMA-D | 7,442 | 91 | 2,542 / 1,087 (3,629) | 5,084 / 2,358 |
| TESS | 5,600 | **2** | 1,600 / 800 (2,400) | 4,000 / 1,600 |
| SAVEE | 480 | 4 | 120 / 120 (240) | 300 / 180 |
| **Combined** | **14,962** | **121** | **4,646 / 2,295 (6,941)** | **10,344 / 4,618** |

## Findings that matter for SP1 design

1. **Scale jump vs. synthetic.** All prior A/B work trained on synthetic data.
   Real pool is ~15k utterances / 121 speakers — enough for an honest
   speaker-independent split.

2. **Class imbalance is ~2:1 stressed:calm** in *both* mappings (more emotion
   categories sit high-arousal). Needs class weighting or balanced sampling;
   raw training will bias toward "stressed". Current `_metrics` already reports
   balanced accuracy, so the harness is ready for this.

3. **narrow drops 54% of the data** (8,021 of 14,962). broad keeps everything
   but folds in ambiguous middles (sad→calm, happy/disgust→stressed) that are
   weak arousal proxies. Open SP1 decision: narrow (cleaner labels, less data)
   vs broad (more data, noisier axis) vs a 3-way design that excludes the
   ambiguous classes explicitly.

4. **Speaker diversity is wildly uneven.** CREMA-D carries the diversity (91
   speakers); **TESS has only 2 speakers (both female, OAF/YAF)** and SAVEE 4
   (all male). A naive pooled split risks speaker leakage from TESS and a
   gender skew. Speaker-independent splits must be assigned per-corpus, and
   TESS should sit entirely on one side of the split (it cannot support its own
   held-out speakers).

5. **Loaders still needed.** Only `ravdess_dataset()` exists. CREMA-D / TESS /
   SAVEE each have a distinct labeling scheme (parsed above) and need a unified
   multi-corpus loader → this is the core SP1 deliverable, pending design
   approval.

_Status: recon only. Implementing the loader / training on this data remains
behind the v2 brainstorm-approval gate._
