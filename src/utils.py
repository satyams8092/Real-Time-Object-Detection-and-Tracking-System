import cv2
import numpy as np

def draw_detections(frame, tracks, trail_viz=None):
    """Draws bounding boxes, IDs and labels for all active tracks."""
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.bbox)
        color = _get_track_color(t.id)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label   = f"ID:{t.id} {t.label}"
        (tw, th), _ = cv2.getTextSize(label,
                                       cv2.FONT_HERSHEY_SIMPLEX,
                                       0.55, 1)
        cv2.rectangle(frame,
                      (x1, y1 - th - 8), (x1 + tw + 6, y1),
                      color, -1)
        cv2.putText(frame, label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, fps, n_tracks, mode):
    """Draws heads-up display with FPS, object count, and active mode."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(frame, 1.0, frame, 0, 0, frame)
    cv2.putText(frame, f"FPS      : {fps:.1f}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 180), 1)
    cv2.putText(frame, f"Tracking : {n_tracks} objects",
                (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 180), 1)
    cv2.putText(frame, f"Mode     : {mode}",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 180), 1)
    return frame


def _get_track_color(track_id):
    np.random.seed(track_id * 13)
    return tuple(int(c) for c in np.random.randint(100, 255, 3))