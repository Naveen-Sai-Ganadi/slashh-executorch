# Per-SNR calibration breakdown — can you trust the confidence when it's noisy?

The pooled score-calibration ECE hides *where* calibration lives. This re-fits per SNR: confidence is only a real probability where the model beats chance.

- noise levels: **5**  ·  global `DEFAULT_TEMPERATURE`=0.41
- ECE grows monotonically with noise: **True**
- floor (-5 dB) trustworthy: **True**
- **Verdict: the *prediction* is reliable across the whole sweep — even at the -5 dB floor accuracy holds at 1.00. But the *confidence number* drifts under-confident with noise: ECE climbs 0.148->0.373 from quiet to floor, and even after the global T=0.41 the floor keeps 0.226 residual ECE vs 0.015 in the quiet. Trust the decision everywhere; treat low-SNR confidence as under-stated (a single global temperature under-corrects at the edge).**

| SNR | n | accuracy | ECE | fitted T | ECE after T | trustworthy |
|---|---|---|---|---|---|---|
| clean | 192 | 1.000 | 0.1479 | 0.41 | 0.0146 | True |
| 20 dB | 192 | 1.000 | 0.1612 | 0.41 | 0.0192 | True |
| 10 dB | 192 | 1.000 | 0.2369 | 0.41 | 0.0819 | True |
| 0 dB | 192 | 0.964 | 0.2926 | 0.41 | 0.1665 | True |
| -5 dB | 192 | 1.000 | 0.3731 | 0.41 | 0.2261 | True |
