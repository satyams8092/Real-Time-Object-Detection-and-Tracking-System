from ultralytics import YOLO
import numpy as np

class YOLOv8Detector:
    """
    Wraps YOLOv8 for real-time object detection.
    Downloads yolov8n.pt (nano) on first run — fastest variant.
    Switch to yolov8s/m/l for better accuracy at cost of speed.
    """
    def __init__(self, model_path="yolov8n.pt",
                 conf_thresh=0.4, iou_thresh=0.45):
        self.model       = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh  = iou_thresh

    def detect(self, frame):
        """
        Runs inference on a single BGR frame.
        Returns list of dicts:
          { 'bbox': [x1,y1,x2,y2], 'conf': float,
            'class_id': int, 'label': str }
        """
        results = self.model(frame,
                             conf=self.conf_thresh,
                             iou=self.iou_thresh,
                             verbose=False)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "bbox"     : [x1, y1, x2, y2],
                "conf"     : float(box.conf[0]),
                "class_id" : int(box.cls[0]),
                "label"    : results.names[int(box.cls[0])]
            })
        return detections