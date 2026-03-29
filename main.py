import cv2
import time
import argparse
from src.detector       import YOLOv8Detector
from src.tracker        import MultiObjectTracker
from src.optical_flow   import KLTOpticalFlow
from src.background_sub import BackgroundSubtractor
from src.trails         import TrailVisualizer
from src.utils          import draw_detections, draw_hud

def run(source, show_flow, show_bgsub, show_trails,
        conf, output_path):

    # --- Input source ---
    cap = cv2.VideoCapture(0 if source == "camera" else source)
    assert cap.isOpened(), f"[!] Cannot open source: {source}"

    # --- Modules ---
    detector  = YOLOv8Detector(conf_thresh=conf)
    tracker   = MultiObjectTracker(iou_thresh=0.3, max_misses=8)
    klt       = KLTOpticalFlow(refresh_interval=30)
    bgsub     = BackgroundSubtractor(method="MOG2")
    trails    = TrailVisualizer(max_trail_len=50)

    # --- Video writer ---
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    prev_time = time.time()
    print("[*] Running — press Q to quit | B: bgsub | F: flow | T: trails")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()

        # 1. Background subtraction layer
        if show_bgsub:
            _, clean_mask, display = bgsub.apply(display)

        # 2. Optical flow layer
        if show_flow:
            display = klt.process(display)

        # 3. YOLOv8 detection
        detections = detector.detect(frame)

        # 4. Multi-object tracking
        active_tracks = tracker.update(detections)

        # 5. Trail visualization
        if show_trails:
            trails.update(active_tracks)
            display = trails.draw(display)

        # 6. Draw bounding boxes + IDs
        display = draw_detections(display, active_tracks)

        # 7. HUD overlay
        now      = time.time()
        fps_val  = 1.0 / (now - prev_time + 1e-6)
        prev_time = now
        mode_str = " | ".join(filter(None, [
            "BGSub"  if show_bgsub  else "",
            "Flow"   if show_flow   else "",
            "Trails" if show_trails else ""
        ])) or "Detection only"
        display = draw_hud(display, fps_val,
                           len(active_tracks), mode_str)

        cv2.imshow("Object Detection & Tracking", display)
        if writer:
            writer.write(display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            show_bgsub  = not show_bgsub
            print(f"[*] BGSub  → {'ON' if show_bgsub  else 'OFF'}")
        elif key == ord('f'):
            show_flow   = not show_flow
            print(f"[*] Flow   → {'ON' if show_flow   else 'OFF'}")
        elif key == ord('t'):
            show_trails = not show_trails
            print(f"[*] Trails → {'ON' if show_trails else 'OFF'}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Object Detection & Tracking — CSE3010 BYOP"
    )
    parser.add_argument("--source",  default="camera",
                        help="'camera' or path to video file")
    parser.add_argument("--flow",    action="store_true",
                        help="Enable KLT optical flow")
    parser.add_argument("--bgsub",   action="store_true",
                        help="Enable background subtraction")
    parser.add_argument("--trails",  action="store_true",
                        help="Enable object trail visualization")
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="YOLO confidence threshold")
    parser.add_argument("--output",  default=None,
                        help="Path to save output video")
    args = parser.parse_args()

    run(args.source, args.flow, args.bgsub,
        args.trails, args.conf, args.output)


if __name__ == "__main__":
    main()