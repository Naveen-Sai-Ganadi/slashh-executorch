# S0 — Model load verification status

Repointed at `datasets/unified/unified.pt` (14,962 samples, [N,1,64,301]).

| # | Model | Track | Load status | Notes |
|---|-------|-------|-------------|-------|
| 1 | StressNet v2 (custom CNN) | A on-device | OK (`model.model.build_model_v2`) | from-scratch on unified data; smoke 0.71 bal-acc @3ep |
| 2 | whisper-tiny encoder (384-d) | A on-device | OK | input_features (1,80,3000) → 384-d; HF cached |
| 3 | yamnet (1024-d) | A on-device | BLOCKED | needs `git+https://github.com/w-hc/torch_audioset.git@e8852c5`; auto-mode classifier denied the git install. Deferred pending approval. |
| 4 | audeering wav2vec2-arousal | B SOTA/teacher | OK | logits (1,3) = arousal/dominance/valence DIRECTLY |
| 5 | emotion2vec_plus_large | B SOTA/teacher | pending | needs funasr (not yet installed) |
| 6 | WavLM-large + SER head | B SOTA/teacher | OK | `microsoft/wavlm-large`; pooled 1024-d embedding |
| 7 | wav2vec2/HuBERT SER | B SOTA/teacher | OK | `superb/hubert-large-superb-er`; logits (1,4) neu/hap/ang/sad |

**Resolved:** `transformers 5.12.1` clashed with qai_hub_models' pins (it needs
`huggingface_hub>=1.5.0`; qai_hub_models needs `<=0.36.2` + `transformers==4.56.2`).
Downgraded to `transformers==4.56.2` + `huggingface_hub 0.36.2`; SOTA models run fine on it.

**Verified:** 5/7 load + emit usable outputs. yamnet blocked on a git-install permission; emotion2vec pending funasr.

## yamnet unblock options
The torch_audioset install is the official qai_hub_models recipe (pinned in
`yamnet/code-gen.yaml`: `git+https://github.com/w-hc/torch_audioset.git`). It was
denied only because the classifier flags any external-git install. To proceed,
the user can approve the install or add a Bash allow-rule. Until then yamnet is
deferred; the other 6 models proceed.
