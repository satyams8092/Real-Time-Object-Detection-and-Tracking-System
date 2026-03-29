import numpy as np
from scipy.optimize import linear_sum_assignment

class KalmanTrack:
    """
    Single object track with a simple Kalman filter.
    State: [x, y, w, h, vx, vy, vw, vh]
    Predicts next position using constant velocity model.
    """
    _id_counter = 0

    def __init__(self, bbox):
        KalmanTrack._id_counter += 1
        self.id       = KalmanTrack._id_counter
        self.bbox     = np.array(bbox, dtype=float)   # [x1,y1,x2,y2]
        self.velocity = np.zeros(4)                    # [vx,vy,vw,vh]
        self.hits     = 1
        self.misses   = 0
        self.label    = ""

    def predict(self):
        """Advance state by one frame using velocity."""
        self.bbox     += self.velocity
        self.misses   += 1
        return self.bbox.copy()

    def update(self, bbox):
        """Correct state with new detection."""
        new_bbox      = np.array(bbox, dtype=float)
        self.velocity = (new_bbox - self.bbox) * 0.5   # EMA velocity
        self.bbox     = new_bbox
        self.hits    += 1
        self.misses   = 0

    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1+x2)/2, (y1+y2)/2)


def iou(boxA, boxB):
    """Intersection over Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]);  yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]);  yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    aA    = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB    = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / float(aA + aB - inter + 1e-6)


class MultiObjectTracker:
    """
    SORT-style tracker:
    1. Predict all existing tracks forward
    2. Match detections to tracks via IoU + Hungarian algorithm
    3. Update matched tracks, create new, delete lost ones
    """
    def __init__(self, iou_thresh=0.3, max_misses=5):
        self.tracks     = []
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses

    def update(self, detections):
        """
        detections : list of dicts with 'bbox' and 'label' keys
        Returns    : list of active KalmanTrack objects
        """
        # Predict
        for t in self.tracks:
            t.predict()

        if not detections:
            self.tracks = [t for t in self.tracks
                           if t.misses <= self.max_misses]
            return self.tracks

        det_bboxes = [d["bbox"]  for d in detections]
        det_labels = [d["label"] for d in detections]

        if self.tracks:
            # Build IoU cost matrix: tracks × detections
            cost = np.zeros((len(self.tracks), len(det_bboxes)))
            for i, t in enumerate(self.tracks):
                for j, db in enumerate(det_bboxes):
                    cost[i, j] = 1 - iou(t.bbox, db)

            # Hungarian assignment
            row_ids, col_ids = linear_sum_assignment(cost)

            matched_t, matched_d = set(), set()
            for r, c in zip(row_ids, col_ids):
                if cost[r, c] < (1 - self.iou_thresh):
                    self.tracks[r].update(det_bboxes[c])
                    self.tracks[r].label = det_labels[c]
                    matched_t.add(r);  matched_d.add(c)

            # New tracks for unmatched detections
            for j, (bbox, label) in enumerate(
                    zip(det_bboxes, det_labels)):
                if j not in matched_d:
                    t       = KalmanTrack(bbox)
                    t.label = label
                    self.tracks.append(t)
        else:
            for bbox, label in zip(det_bboxes, det_labels):
                t       = KalmanTrack(bbox)
                t.label = label
                self.tracks.append(t)

        # Remove lost tracks
        self.tracks = [t for t in self.tracks
                       if t.misses <= self.max_misses]
        return self.tracks