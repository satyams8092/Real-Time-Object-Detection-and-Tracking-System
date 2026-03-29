# 🎯 Real-Time Object Detection & Tracking System

> A computer vision pipeline built for **CSE3010 — Computer Vision (VIT Bhopal University)**  
> BYOP Submission | YOLOv8 · KLT Optical Flow · MOG2 Background Subtraction · Multi-Object Tracking

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Demo](#-demo)
- [Features](#-features)
- [Syllabus Coverage](#-syllabus-coverage)
- [Project Structure](#-project-structure)
- [System Requirements](#-system-requirements)
- [Local Setup — Step by Step](#-local-setup--step-by-step)
- [How to Run](#-how-to-run)
- [CLI Reference](#-cli-reference)
- [Keyboard Controls](#-keyboard-controls)
- [Module Descriptions](#-module-descriptions)
- [Results & Output](#-results--output)
- [Tech Stack](#-tech-stack)
- [References](#-references)

---

## 📖 Project Overview

This project implements a **real-time object detection and tracking system** using state-of-the-art computer vision techniques. It combines:

- **YOLOv8** for fast, accurate multi-class object detection
- **Kalman Filter + Hungarian Algorithm** for robust multi-object tracking (SORT-style)
- **KLT (Kanade-Lucas-Tomasi)** sparse optical flow for feature-level motion estimation
- **MOG2 Background Subtraction** (Mixture of Gaussians) for foreground segmentation
- **Fading Trail Visualization** to show object movement history over time

The system works on both **live webcam feeds** and **pre-recorded video files**, with real-time feature toggling via keyboard.

---

## 🎬 Demo

```
python main.py --source camera --flow --bgsub --trails
```

> 📹 A sample demo video (`demo.mp4`) is included in the `outputs/` folder.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔍 **YOLOv8 Detection** | Detects 80 COCO object classes in real time |
| 🔢 **Multi-Object Tracking** | Assigns persistent IDs across frames using Kalman + Hungarian |
| 🌊 **KLT Optical Flow** | Tracks feature points using pyramidal Lucas-Kanade |
| 🎭 **Background Subtraction** | MOG2 foreground mask with morphological cleanup |
| 🌈 **Object Trails** | Fading colored trails show each object's movement path |
| 📺 **Dual Input** | Supports webcam (live) and video file input |
| 💾 **Video Export** | Save annotated output as `.mp4` |
| ⌨️ **Live Toggle** | Toggle features on/off in real time using keyboard |
| 📊 **HUD Overlay** | Shows live FPS, object count, and active mode |

---

## 📚 Syllabus Coverage

| Module | Concept | Implementation |
|---|---|---|
| Module 1 | Color space conversion, Morphology | `background_sub.py` |
| Module 3 | Shi-Tomasi corner detection, Feature extraction | `optical_flow.py` |
| Module 3 | Object detection (YOLO = deep CNN features) | `detector.py` |
| Module 4 | Mixture of Gaussians (MOG2) | `background_sub.py` |
| Module 4 | KLT Tracker, Optical Flow | `optical_flow.py` |
| Module 4 | Kalman Filter, Hungarian Algorithm | `tracker.py` |
| Module 4 | Spatio-Temporal Analysis | `trails.py` |
| Experiment 9 | Optical flow method | `optical_flow.py` |
| Experiment 10 | Object detection & tracking from video | `main.py` |
| Experiment 12 | Object detection from dynamic background | `background_sub.py` |

---

## 📁 Project Structure

```
object-detection-tracking/
│
├── assets/
│   └── coco.txt                  # 80 COCO class names
│
├── src/
│   ├── detector.py               # YOLOv8 detection wrapper
│   ├── tracker.py                # Multi-object tracker (Kalman + Hungarian)
│   ├── optical_flow.py           # KLT sparse optical flow (Lucas-Kanade)
│   ├── background_sub.py         # MOG2 / KNN background subtraction
│   ├── trails.py                 # Fading object trail visualization
│   └── utils.py                  # Drawing helpers, HUD, color palette
│
├── outputs/                      # Saved output videos go here
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 💻 System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| **OS** | Windows 10 / Ubuntu 20.04 / macOS 11 | Ubuntu 22.04 / Windows 11 |
| **Python** | 3.8 | 3.10+ |
| **RAM** | 4 GB | 8 GB+ |
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **GPU** | Optional | NVIDIA GPU (CUDA) for faster inference |
| **Webcam** | Any USB / built-in | HD 1080p |
| **Storage** | ~500 MB (model + deps) | 1 GB+ |

> ⚠️ **Note:** YOLOv8 runs on CPU by default. With an NVIDIA GPU + CUDA, inference is 5–10× faster.

---

## 🛠️ Local Setup — Step by Step

### Step 1 — Clone the Repository

```bash
git clone https://github.com/{your-username}/object-detection-tracking.git
cd object-detection-tracking
```

---

### Step 2 — Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> ✅ You should see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**`requirements.txt` contents:**
```
ultralytics
opencv-python
numpy
scipy
```

> ⏳ First install may take 2–5 minutes. `ultralytics` downloads YOLOv8 weights (~6 MB for nano model) automatically on first run.

---

### Step 4 — Verify Installation

```bash
python -c "import cv2; import ultralytics; import numpy; print('All dependencies OK!')"
```

Expected output:
```
All dependencies OK!
```

---

### Step 5 — (Optional) GPU Setup for NVIDIA

If you have an NVIDIA GPU, install PyTorch with CUDA for faster inference:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify GPU is detected:
```bash
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

---

## ▶️ How to Run

### 🎥 Run on Webcam (Real-Time)

```bash
# Basic — detection only
python main.py --source camera

# All features enabled
python main.py --source camera --flow --bgsub --trails

# With custom confidence threshold
python main.py --source camera --flow --bgsub --trails --conf 0.5
```

---

### 📂 Run on a Video File

```bash
# Basic detection on a video file
python main.py --source path/to/your/video.mp4

# All features on video
python main.py --source path/to/your/video.mp4 --flow --bgsub --trails

# Save annotated output
python main.py --source path/to/your/video.mp4 --flow --bgsub --trails --output outputs/result.mp4
```

---

### 💾 Save Output Video

```bash
python main.py --source camera --flow --bgsub --trails --output outputs/demo.mp4
```

> Output is saved to the `outputs/` folder as an `.mp4` file.

---

## 📋 CLI Reference

| Argument | Type | Default | Description |
|---|---|---|---|
| `--source` | `str` | `camera` | Input: `"camera"` or path to video file |
| `--flow` | `flag` | `False` | Enable KLT optical flow visualization |
| `--bgsub` | `flag` | `False` | Enable MOG2 background subtraction |
| `--trails` | `flag` | `False` | Enable fading object trail visualization |
| `--conf` | `float` | `0.4` | YOLOv8 detection confidence threshold (0.0–1.0) |
| `--output` | `str` | `None` | Path to save annotated output video |

---

## ⌨️ Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `B` | Toggle Background Subtraction ON / OFF |
| `F` | Toggle Optical Flow ON / OFF |
| `T` | Toggle Object Trails ON / OFF |

> You can toggle features **live** while the system is running — no need to restart!

---

## 🧩 Module Descriptions

### `src/detector.py` — YOLOv8 Detector
Wraps the Ultralytics YOLOv8 model for real-time inference. Returns bounding boxes, confidence scores, class IDs, and class labels for every detected object. Uses `yolov8n.pt` (nano) by default for speed; swap to `yolov8s/m/l` for higher accuracy.

### `src/tracker.py` — Multi-Object Tracker
Implements a SORT-style tracker combining:
- **Kalman Filter** — predicts object position between frames using a constant velocity model
- **IoU cost matrix** — measures similarity between predicted and detected boxes
- **Hungarian Algorithm** (`scipy.optimize.linear_sum_assignment`) — globally optimal assignment of detections to tracks

### `src/optical_flow.py` — KLT Optical Flow
Implements sparse optical flow using OpenCV's `calcOpticalFlowPyrLK`:
- **Shi-Tomasi** corner detection to find good features to track
- **Pyramidal Lucas-Kanade** to track those features across frames
- Refreshes feature points every 30 frames to maintain tracking quality
- Draws motion arrows on the frame

### `src/background_sub.py` — Background Subtraction
Uses OpenCV's `createBackgroundSubtractorMOG2`:
- Models each pixel as a **Mixture of Gaussians** (ties directly to Module 4)
- Removes shadow pixels with thresholding
- Applies **morphological opening** (erosion + dilation) to remove noise
- Returns a green-tinted foreground overlay

### `src/trails.py` — Trail Visualizer
Maintains a deque of center positions for each tracked object ID:
- Each trail is colored uniquely per object ID
- Trail opacity and thickness fade with age (newer = thicker/brighter)
- Dead tracks are automatically pruned

### `src/utils.py` — Drawing Utilities
Provides bounding box rendering with label backgrounds and a HUD overlay showing live FPS, active object count, and enabled feature modes.

---

## 📊 Results & Output

The system outputs annotated video frames showing:

- **Colored bounding boxes** around each detected object
- **Track ID + class label** above each box
- **Fading trails** showing movement history
- **Optical flow arrows** showing feature-level motion vectors
- **Green foreground mask** overlay from background subtraction
- **HUD** showing FPS, object count, and active modules

---

## 🔧 Tech Stack

| Component | Library / Model |
|---|---|
| Object Detection | [YOLOv8](https://github.com/ultralytics/ultralytics) (Ultralytics) |
| Computer Vision | [OpenCV](https://opencv.org/) 4.x |
| Numerical Computing | [NumPy](https://numpy.org/) |
| Hungarian Algorithm | [SciPy](https://scipy.org/) (`linear_sum_assignment`) |
| Language | Python 3.8+ |

---

## 📖 References

1. Redmon, J. et al. — *You Only Look Once: Unified, Real-Time Object Detection* (CVPR 2016)
2. Bewley, A. et al. — *Simple Online and Realtime Tracking (SORT)* (ICIP 2016)
3. Lucas, B. D. & Kanade, T. — *An Iterative Image Registration Technique with an Application to Stereo Vision* (IJCAI 1981)
4. Stauffer, C. & Grimson, W. — *Adaptive Background Mixture Models for Real-Time Tracking* (CVPR 1999)
5. Richard Szeliski — *Computer Vision: Algorithms and Applications* — Springer, 2011 *(Course Textbook)*
6. Forsyth, D. A. & Ponce, J. — *Computer Vision: A Modern Approach* — Pearson, 2003 *(Course Textbook)*

---

## 👤 Author

**Satyam Shah**  
Registration No: 23BAI11021  
Course: CSE3010 — Computer Vision  
VIT Bhopal University, Madhya Pradesh

---

## 📄 License

This project is submitted as part of the BYOP component of CSE3010 at VIT Bhopal University.  
All code is original work by the author.

---

> ⭐ If you found this helpful, consider starring the repository!
