"""StressNet — tiny log-mel CNN classifier.

Input : log-mel tensor [B, 1, N_MELS, N_FRAMES]  (see audio_config)
Output: stress score   [B, 1] in [0, 1]   (1.0 == high arousal / "stressed")

Deliberately small so it (a) runs in well under real-time, (b) quantizes
cleanly to INT8, and (c) lowers to XNNPACK now and the QNN/Hexagon NPU later.
Only conv/bn/relu/pool/linear/sigmoid — all delegate-friendly ops; feature
extraction lives outside the graph (computed on-device).
"""

from __future__ import annotations

import torch
from torch import nn

from .audio_config import N_FRAMES, N_MELS


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))


class StressNet(nn.Module):
    """Binary calm-vs-stressed classifier over a fixed log-mel window."""

    def __init__(self, channels: tuple[int, int, int] = (16, 32, 64)) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            ConvBlock(1, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)   # -> [B, c3, 1, 1]
        self.head = nn.Linear(c3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)           # [B, c3]
        logit = self.head(x)                  # [B, 1]
        return torch.sigmoid(logit)           # [B, 1] in [0, 1]


def example_input(batch: int = 1) -> torch.Tensor:
    """A representative input tensor for export / smoke tests."""
    return torch.randn(batch, 1, N_MELS, N_FRAMES)


def build_model(weights: str | None = None) -> StressNet:
    """Construct StressNet in eval mode, optionally loading a checkpoint."""
    model = StressNet()
    if weights is not None:
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model
