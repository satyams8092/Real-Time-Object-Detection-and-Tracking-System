import cv2
import numpy as np

class KLTOpticalFlow:
    """
    Implements KLT (Kanade-Lucas-Tomasi) sparse optical flow.
    Tracks good features (corners) across frames using
    pyramidal Lucas-Kanade method — directly from Module 4.
    """

    # Shi-Tomasi corner detection parameters
    FEATURE_PARAMS = dict(maxCorners=200,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

    # Lucas-Kanade optical flow parameters
    LK_PARAMS = dict(winSize=(15, 15),
                     maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS |
                                cv2.TERM_CRITERIA_COUNT,
                                10, 0.03))

    def __init__(self, refresh_interval=30):
        self.prev_gray    = None
        self.prev_pts     = None
        self.flow_vectors = []             # List of (start, end) tuples
        self.frame_count  = 0
        self.refresh_interval = refresh_interval  # Re-detect corners every N frames

    def process(self, frame):
        """
        Computes sparse optical flow for the current frame.
        Returns frame with flow arrows drawn on it.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = frame.copy()

        if (self.prev_gray is None or
                self.prev_pts is None or
                self.frame_count % self.refresh_interval == 0):
            # Detect new feature points to track
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, mask=None, **self.FEATURE_PARAMS
            )
            self.prev_gray   = gray.copy()
            self.frame_count = 0

        if self.prev_pts is not None and len(self.prev_pts) > 0:
            # Track points using pyramidal LK
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray,
                self.prev_pts, None,
                **self.LK_PARAMS
            )
            if next_pts is not None:
                good_new  = next_pts[status == 1]
                good_prev = self.prev_pts[status == 1]

                self.flow_vectors = list(zip(
                    good_prev.reshape(-1, 2),
                    good_new.reshape(-1, 2)
                ))

                # Draw flow arrows
                for prev_pt, next_pt in self.flow_vectors:
                    px, py = map(int, prev_pt)
                    nx, ny = map(int, next_pt)
                    magnitude = np.sqrt((nx-px)**2 + (ny-py)**2)
                    if magnitude > 1.0:             # filter static points
                        cv2.arrowedLine(output,
                                        (px, py), (nx, ny),
                                        (0, 140, 255), 1,
                                        tipLength=0.3)
                        cv2.circle(output, (nx, ny),
                                   2, (0, 255, 140), -1)

                self.prev_pts  = good_new.reshape(-1, 1, 2)

        self.prev_gray = gray.copy()
        self.frame_count += 1
        return output