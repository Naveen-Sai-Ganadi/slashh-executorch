"""Calming-intervention policy — host mirror of the on-device decision (M9).

The detector says *whether this window is stressed*; this policy decides *whether
to interrupt the user with a calming cue*. That decision is pure stream logic and
lives here so it is unit-testable on the host, exactly as :mod:`model.pipeline`
mirrors the Kotlin stream pipeline. The device intervention layer makes the same
call.

Two product rules drive it (M9 acceptance: "sustained stress triggers a calming
cue; intervention is dismissible and rate-limited — no nagging"):

  * **Sustained** — fire only after stress holds for ``sustain_windows``
    consecutive voiced windows; any calm or unvoiced window resets the streak, so
    a single noisy blip never triggers a prompt.
  * **No nagging** — after firing (or a user ``dismiss()``) the policy enters a
    ``cooldown_windows`` cooldown during which it will not fire again.

    from model.intervention import InterventionPolicy, InterventionConfig
    policy = InterventionPolicy(InterventionConfig())
    if policy.update(state).fire:
        show_calming_cue()
    # on user dismissal: policy.dismiss()

Host-only; no device, no AI Hub token.
"""

from __future__ import annotations

from dataclasses import dataclass

from .detector import StressState


@dataclass(frozen=True)
class InterventionConfig:
    """Tuning for the calming cue.

    ``sustain_windows`` consecutive stressed windows arm the cue; after it fires
    (or is dismissed) it stays quiet for ``cooldown_windows`` windows. Windows are
    one detector hop apart (``HOP_SECONDS``), so the defaults are ~3 s of sustained
    stress and ~20 s of quiet between prompts.
    """

    sustain_windows: int = 3
    cooldown_windows: int = 20


@dataclass(frozen=True)
class InterventionDecision:
    """Outcome of one window. ``fire`` is True only on the window that triggers."""

    fire: bool
    sustained_windows: int
    in_cooldown: bool


class InterventionPolicy:
    """Decide when to surface a calming cue from a stream of :class:`StressState`."""

    def __init__(self, config: InterventionConfig | None = None) -> None:
        self.config = config or InterventionConfig()
        self._streak = 0
        self._cooldown = 0

    @property
    def in_cooldown(self) -> bool:
        """True while the policy is rate-limited and will not fire."""
        return self._cooldown > 0

    def update(self, state: StressState) -> InterventionDecision:
        """Feed one window; returns whether to fire the calming cue now."""
        # Cooldown ticks down once per processed window.
        if self._cooldown > 0:
            self._cooldown -= 1

        if self.in_cooldown:
            # Rate-limited: hold the streak at zero so a fresh episode is required
            # once the cooldown fully elapses (no nagging while still in cooldown).
            self._streak = 0
            return InterventionDecision(False, 0, in_cooldown=True)

        # Sustained-stress streak: any calm / unvoiced window breaks it.
        if state.voiced and state.stressed:
            self._streak += 1
        else:
            self._streak = 0

        fire = self._streak >= self.config.sustain_windows
        if fire:
            self._cooldown = self.config.cooldown_windows
            self._streak = 0  # require a fresh episode before firing again

        return InterventionDecision(
            fire=fire, sustained_windows=self._streak, in_cooldown=self.in_cooldown
        )

    def dismiss(self) -> None:
        """User dismissed the cue: open the cooldown so we don't nag."""
        self._cooldown = self.config.cooldown_windows
        self._streak = 0

    def reset(self) -> None:
        """Clear all state (e.g. on a new session)."""
        self._streak = 0
        self._cooldown = 0
