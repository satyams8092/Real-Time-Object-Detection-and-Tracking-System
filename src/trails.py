import cv2
import numpy as np
from collections import defaultdict, deque

class TrailVisualizer:
    """
    Maintains a position history for each tracked object ID
    and draws a fading color trail showing movement history.
    """
    def __init__(self, max_trail_len=40):
        self.trails      = defaultdict(lambda: deque(maxlen=max_trail_len))
        self.color_cache = {}

    def _get_color(self, track_id):
        """Assigns a consistent unique BGR color to each track ID."""
        if track_id not in self.color_cache:
            np.random.seed(track_id * 37)
            self.color_cache[track_id] = tuple(
                int(c) for c in np.random.randint(80, 255, 3)
            )
        return self.color_cache[track_id]

    def update(self, tracks):
        """Add current center position of each active track."""
        active_ids = set()
        for t in tracks:
            cx, cy = map(int, t.center())
            self.trails[t.id].append((cx, cy))
            active_ids.add(t.id)
        # Prune trails for dead tracks
        dead = [tid for tid in self.trails if tid not in active_ids]
        for tid in dead:
            del self.trails[tid]

    def draw(self, frame):
        """
        Draws fading trails on the frame.
        Older points are thinner and more transparent.
        """
        overlay = frame.copy()
        for track_id, pts in self.trails.items():
            color    = self._get_color(track_id)
            pts_list = list(pts)
            n        = len(pts_list)
            for i in range(1, n):
                alpha     = i / n              # fade: older = more transparent
                thickness = max(1, int(3 * alpha))
                faded_col = tuple(int(c * alpha) for c in color)
                cv2.line(overlay,
                         pts_list[i-1], pts_list[i],
                         faded_col, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        return frame