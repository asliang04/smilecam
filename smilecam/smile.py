import math


class SmileTrigger:
    """Turns a per-frame "is someone smiling" signal into one event per smile.

    Timing is in seconds rather than frames so it behaves the same on a fast
    laptop GPU and a slow CPU-only device.
    """

    def __init__(self, hold: float, cooldown: float):
        self.hold = hold
        self.cooldown = cooldown
        self._started: float | None = None  # when the current smile began
        self._fired = False  # already captured the current smile
        self._last_fire = -math.inf

    def update(self, smiling: bool, now: float) -> bool:
        if not smiling:
            self._started = None
            self._fired = False
            return False
        if self._started is None:
            self._started = now
        if self._fired or now - self._started < self.hold or now - self._last_fire < self.cooldown:
            return False
        self._fired = True
        self._last_fire = now
        return True
