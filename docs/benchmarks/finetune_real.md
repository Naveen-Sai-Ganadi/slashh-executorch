# Fine-tune on real speech (RAVDESS + EMO-DB)

StressNet fine-tuned from **assets/stress_model.pt** on real emotional speech mapped to the arousal axis, under a **speaker-independent** split (1489 train clips; 486 validation clips from held-out speakers). Stressed (high-arousal) fraction of the pool: **0.64**.

Held-out (unseen speakers) is the number that matters:

- **accuracy: 0.687**
- precision: 0.982 · recall: 0.525 · f1: 0.685
- confusion: tp=165 fp=3 tn=169 fn=149
- per-corpus accuracy: emodb: 0.810 · ravdess: 0.644

At 0.5 the model is conservative (high precision, low recall). Selecting the operating point on the **train** split gives threshold **0.03**; applied blind to held-out:

- **accuracy: 0.879**
- precision: 0.881 · recall: 0.939 · f1: 0.909
- confusion: tp=295 fp=40 tn=132 fn=19
- per-corpus accuracy: emodb: 0.929 · ravdess: 0.861

Train-split accuracy (same speakers, for reference): 0.692

Held-out speakers: emodb:11, emodb:16, ravdess:03, ravdess:04, ravdess:06, ravdess:08, ravdess:13, ravdess:21

Exported `.pte`: 97.9 KB
