from collections import Counter

from .objects import Detection


class SeenCounter:
    """Counts distinct tracked objects per label over the whole session.

    A track is counted once it has lasted `min_age` seconds, which skips brief false
    detections. Someone who leaves the frame and comes back later gets a new track ID,
    so they are counted again; telling them apart would need face or person re-identification.
    """

    def __init__(self, min_age: float):
        self.min_age = min_age
        self.totals: Counter[str] = Counter()
        self._first_seen: dict[int, float] = {}
        self._counted: set[int] = set()

    def update(self, detections: list[Detection], now: float) -> None:
        for det in detections:
            tid = det.track_id
            if tid is None or tid in self._counted:
                continue
            if now - self._first_seen.setdefault(tid, now) >= self.min_age:
                del self._first_seen[tid]
                self._counted.add(tid)
                self.totals[det.label] += 1
