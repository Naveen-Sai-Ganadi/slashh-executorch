# StressNet failure mode under noise

Same waveform-noise SNR sweep as the robustness curve, split by error direction. `precision` falls when the model **false-alarms** (FP); `recall` falls when it **misses stress** (FN). The dominant direction is only named when one error type is at least 2× the other.

- **First failure leans: misses stress (false negatives — detector goes silent).** Tune the detector gate accordingly: a silent model wants a lower stress threshold / faster attack; a trigger-happy one wants a higher threshold / more smoothing.

| SNR | accuracy | precision | recall | FP | FN | direction |
|---|---|---|---|---|---|---|
| clean | 1.000 | 1.000 | 1.000 | 0 | 0 | balanced |
| 20 dB | 1.000 | 1.000 | 1.000 | 0 | 0 | balanced |
| 10 dB | 0.953 | 0.914 | 1.000 | 6 | 0 | false_alarms |
| 0 dB | 0.922 | 1.000 | 0.844 | 0 | 10 | misses_stress |
| -5 dB | 0.500 | 0.000 | 0.000 | 0 | 64 | misses_stress |
| -10 dB | 0.500 | 0.000 | 0.000 | 0 | 64 | misses_stress |
