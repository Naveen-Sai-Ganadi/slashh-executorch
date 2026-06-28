# Slashh — Documentation

On-device voice stress detection that fuses **WavLM** (how you speak) and **Whisper-Tiny** (what
you say), both running on the **Snapdragon Hexagon NPU**, into one confident, private,
network-free reading.

> New here? Start with the **[README](../README.md)** — it gets the app running on a device
> step-by-step. This hub is the deeper material: how it works, how the NPU runtime is wired, how
> the models were trained, and the full log of everything we tried.

## Contents

| Doc | What's inside |
|-----|---------------|
| **[Architecture](architecture.md)** | The full signal pipeline, every component, every threshold, and *why* each is where it is. The companion to the diagram. |
| **[NPU runtime](npu-runtime.md)** | How **two** models run on the Hexagon NPU out-of-process (the SELinux story), the file-channel protocol, obtaining/building the QNN context binaries, the memory + FUSE gotchas, and how to bring both daemons up after a boot. |
| **[Models](models.md)** | The text stress classifier (Dreaddit), the fusion perceptron, the Whisper-Tiny export, and the StressNet/WavLM audio models — how each is trained/exported, and how to retune the fusion. |
| **[Research journey](research-journey.md)** | The story: the idea, the branches we tried, the dead-ends and the fixes — RAVDESS saturation, the in-app QNN crash, the 614 MB OOM, the FUSE read race, and the fusion-calibration saga. |

## The architecture at a glance

![Architecture](stress-architecture.svg)

## One-paragraph summary

A foreground service captures the mic at 16 kHz and scores a 3-second window every second. A
**VAD** gate skips silence. **WavLM** (audio) scores vocal arousal on the NPU; **Whisper-Tiny**
(ASR) transcribes on the NPU and a tiny **Dreaddit-trained text classifier** scores the words. A
**symmetric fusion perceptron** combines the two scores (it only calls "stressed" when *both*
audio and text are jointly elevated), the result is EMA-smoothed and run through a hysteresis
latch, and sustained stress fires an on-device relief nudge. No audio, transcript, or score ever
leaves the phone — there is no `INTERNET` permission and no LLM anywhere in the loop.

## Historical / model-research docs

Earlier design notes and benchmarks from the v1 StressNet work live alongside this hub:

- [`plan.md`](plan.md) — the original design plan.
- [`benchmarks/`](benchmarks/) — StressNet robustness / envelope / production benchmarks.
- [`v2/`](v2/) — v2 notes incl. the WavLM-on-NPU proof.
