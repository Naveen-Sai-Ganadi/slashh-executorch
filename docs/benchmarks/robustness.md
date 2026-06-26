# StressNet noise robustness

White Gaussian noise is added to each waveform at the SNR below, then features are re-extracted and the model evaluated. The **operating floor** is the first SNR (clean→noisy) where accuracy < 0.80.

- **operating floor: 20 dB**

| SNR | accuracy | f1 | n |
|---|---|---|---|
| clean | 1.000 | 1.000 | 128 |
| 30 dB | 1.000 | 1.000 | 128 |
| 20 dB | 0.680 | 0.757 | 128 |
| 10 dB | 0.500 | 0.667 | 128 |
| 0 dB | 0.500 | 0.667 | 128 |
| -10 dB | 0.500 | 0.667 | 128 |
