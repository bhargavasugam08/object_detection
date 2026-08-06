# Object Detection — Real-time Tracking and Alerts

A simple real-time object detection and multi-object tracking demo using YOLO and DeepSORT.

TL;DR
- Run `python main.py` to start webcam-based detection and tracking.
- Provides on-screen bounding boxes, track IDs, simple distance estimation (based on an assumed object width), and optional speech alerts.

Demo
- Start the webcam demo: `python main.py` (press `q` to quit).

Highlights
- Real-time detection + tracking using YOLO model weights included (yolov8n.pt) and DeepSort for persistent IDs.
- Lightweight UI overlay showing active tracks, detected classes, FPS, and simple distance estimation.

Tech stack
- Python, OpenCV, NumPy, DeepSORT (deep_sort_realtime), PyTorch/Ultralytics (for YOLO weights), pyttsx3 for speech.

Quick start
1. Clone the repo:
   git clone https://github.com/bhargavasugam08/object_detection.git
2. Create a virtual environment and activate it:
   python -m venv venv && source venv/bin/activate  # macOS / Linux
   venv\Scripts\activate  # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Run the demo (webcam):
   python main.py

If you prefer to run on a video file, modify `cap = cv2.VideoCapture(0)` in `main.py` or add a small wrapper to pass a file path.

Repository structure (important files)
- main.py             — main demo and dashboard
- webcam.py           — helper to run webcam demo
- detector.py         — detection logic (model loading + inference)
- speech.py           — text-to-speech helper
- coco.names          — COCO class names
- yolov3.cfg          — YOLO config (if used)
- yolov8n.pt          — small YOLOv8 weights (included for convenience)

How I contributed
- Implemented the detection pipeline, integrated DeepSort for tracking, added visualization overlays and speech alerts, and packaged model weights for quick demos.

Notes & next steps
- Consider removing large weights from the repo and hosting them externally (or use Git LFS) if you plan to keep developing this.
- Add a Dockerfile to containerize the demo and make it easier for recruiters to run.
- Add a short demo GIF or a 60s video showing the pipeline in action.

License
This project is licensed under the MIT License — see LICENSE.

Contact
- GitHub: https://github.com/bhargavasugam08
