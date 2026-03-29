import cv2
import numpy as np

class BackgroundSubtractor:
    """
    Wraps OpenCV's MOG2 background subtractor.
    MOG2 models each pixel as a Mixture of Gaussians —
    directly tied to Module 4 (Mixture of Gaussians, Motion Analysis).

    Also provides KNN-based subtractor as an alternative.
    """
    def __init__(self, method="MOG2",
                 history=500, var_threshold=50,
                 detect_shadows=True):
        if method == "MOG2":
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=history,
                varThreshold=var_threshold,
                detectShadows=detect_shadows
            )
        elif method == "KNN":
            self.subtractor = cv2.createBackgroundSubtractorKNN(
                history=history,
                dist2Threshold=400,
                detectShadows=detect_shadows
            )
        # Morphological kernel to clean up noise
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)
        )

    def apply(self, frame, learning_rate=-1):
        """
        Applies background subtraction to a frame.
        Returns:
          fg_mask  : binary foreground mask (uint8)
          clean    : mask after morphological cleaning
          colored  : green-tinted foreground overlay on frame
        """
        fg_mask = self.subtractor.apply(
            frame, learningRate=learning_rate
        )
        # Remove shadows (gray pixels → 0)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255,
                                    cv2.THRESH_BINARY)
        # Morphological open (erode then dilate) — removes noise
        clean = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                  self.kernel, iterations=2)
        # Colored overlay for visualization
        colored        = frame.copy()
        green_layer    = np.zeros_like(frame)
        green_layer[:, :, 1] = clean   # green channel = mask
        colored = cv2.addWeighted(colored, 0.7,
                                   green_layer, 0.3, 0)
        return fg_mask, clean, colored

    def get_motion_regions(self, clean_mask, min_area=500):
        """
        Finds bounding boxes of significant motion regions
        from the cleaned foreground mask.
        Returns list of [x1, y1, x2, y2] boxes.
        """
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append([x, y, x+w, y+h])
        return boxes