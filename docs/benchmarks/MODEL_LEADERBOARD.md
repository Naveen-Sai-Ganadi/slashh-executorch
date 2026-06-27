# Model leaderboard — held-out-speaker comparison

Balanced accuracy on the unified speaker-independent val split. INT8 columns are on-device `.pte` parity (blank = not yet exported).

| model | kind | map | val bal-acc | INT8 bal-acc | INT8 agree | INT8 KB |
|---|---|---|---|---|---|---|
| `whisper_narrow` | frozen whisper + head | narrow | 0.902 | — | — | — |
| `stressnet_v2_narrow` | custom CNN (v2) | narrow | 0.829 | 0.866 | 0.998 | 260 |
| `whisper_broad` | frozen whisper + head | broad | 0.822 | — | — | — |
| `stressnet_v2_broad` | custom CNN (v2) | broad | 0.767 | 0.791 | 0.996 | 260 |
| `stressnet_v1_broad` | custom CNN (v1) | broad | 0.753 | 0.781 | 0.957 | 34 |

**Best so far:** `whisper_narrow` (frozen whisper + head) at 0.902 bal-acc.
