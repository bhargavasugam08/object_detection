# Object Detection — Real-time Tracking and Alerts

A simple real-time object detection and multi-object tracking demo using YOLO and DeepSORT, built in Python. This project shows how to run detection on a webcam or video file, visualize tracked objects with persistent IDs, estimate simple distances, and optionally announce alerts via speech.

## Highlights
- Real-time object detection + multi-object tracking using YOLO + DeepSORT.
- Lightweight overlay showing bounding boxes, track IDs, class labels, FPS, and simple distance estimation (assumes a fixed object width).
- Optional text-to-speech alerts for detected objects or proximity events.
- Small footprint demo intended for learning, prototyping, and demos.

## Demo
- Start the webcam demo:
  python main.py
  (press `q` to quit)

## Features
- YOLO-based detection (weights included for quick demo).
- DeepSORT for persistent tracking IDs.
- On-screen overlays (class, ID, confidence, FPS).
- Simple distance estimation from bounding box size.
- Optional speech alerts via pyttsx3.

## Tech stack
- Python, OpenCV, NumPy
- Ultralytics / PyTorch (YOLO)
- DeepSORT (deep_sort_realtime)
- pyttsx3 for text-to-speech

## Requirements
- Python 3.8+
- See requirements.txt for exact package versions.

## Quick start
1. Clone the repo:
   git clone https://github.com/bhargavasugam08/object_detection.git
2. Create a virtual environment and activate it:
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate     # Windows (PowerShell/CMD)
3. Install dependencies:
   pip install -r requirements.txt
4. Run the demo (webcam):
   python main.py

## Run on a video file
- To run on a video file, modify the video source in main.py (change `cv2.VideoCapture(0)` to `cv2.VideoCapture("path/to/video.mp4")`) or add a CLI wrapper to pass a file path.

## Configuration & Quick Notes
- Model weights: yolov8n.pt (small weight) is included for convenience. For better accuracy swap in larger YOLO weights and update the config.
- Distance estimation: implemented as a simple heuristic that assumes a known average object width — not for production-grade range finding.
- Speech alerts: enabled/disabled via speech.py; requires pyttsx3 and appropriate system TTS backend.

## Repository structure (important files)
- main.py           — main demo and dashboard
- webcam.py         — helper for webcam usage
- detector.py       — detection logic (model loading + inference)
- deep_sort.py / deep_sort_realtime — tracking integration (depending on implementation)
- speech.py         — text-to-speech helper
- coco.names        — COCO class names
- requirements.txt  — Python dependencies
- yolov8n.pt        — small YOLOv8 weights (included)

## Recommendations / Next steps
- Move large model weights to external hosting (or enable Git LFS) if you plan to keep actively developing.
- Add a Dockerfile to simplify running the demo across environments.
- Add a short demo GIF or 60s video in the repository to showcase the pipeline.

## Contribution
Contributions, bug reports, and suggestions are welcome — open an issue or submit a pull request.

## Contact
- GitHub: https://github.com/bhargavasugam08
