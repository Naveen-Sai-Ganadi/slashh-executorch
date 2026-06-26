# StressNet failure mode under noise

Same waveform-noise SNR sweep as the robustness curve, split by error direction. `precision` falls when the model **false-alarms** (FP); `recall` falls when it **misses stress** (FN). The dominant direction is only named when one error type is at least 2× the other.

- **First failure leans: false alarms (false positives — detector cries wolf).** Tune the detector gate accordingly: a silent model wants a lower stress threshold / faster attack; a trigger-happy one wants a higher threshold / more smoothing.

| SNR | accuracy | precision | recall | FP | FN | direction |
|---|---|---|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 | 0 | 0 | balanced |
| 10 dB | 1.000 | 1.000 | 1.000 | 0 | 0 | balanced |
| 0 dB | 0.961 | 0.927 | 1.000 | 5 | 0 | false_alarms |
| -5 dB | 1.000 | 1.000 | 1.000 | 0 | 0 | balanced |
| -10 dB | 1.000 | 1.000 | 1.000 | 0 | 0 | balanced |
| -15 dB | 0.641 | 0.582 | 1.000 | 46 | 0 | false_alarms |
| -20 dB | 0.500 | 0.500 | 1.000 | 64 | 0 | false_alarms |
