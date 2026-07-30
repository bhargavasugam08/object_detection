import cv2
import time
import numpy as np
from utils.detector import ObjectDetector
from utils.speech import speak
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Config ─────────────────────────────────────────────────────────────────
SPEAK_INTERVAL     = 3
FOCAL_LENGTH       = 615
KNOWN_OBJECT_WIDTH = 0.5
MAX_TRACK_AGE      = 30
# ───────────────────────────────────────────────────────────────────────────

def get_color(class_id):
    np.random.seed(class_id)
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

def estimate_distance(box_width):
    if box_width == 0:
        return None
    distance = (KNOWN_OBJECT_WIDTH * FOCAL_LENGTH) / box_width
    return round(distance, 1)

# Initialize
detector      = ObjectDetector()
tracker       = DeepSort(max_age=MAX_TRACK_AGE)
cap           = cv2.VideoCapture(0)
track_history = {}

if not cap.isOpened():
    print("[Error] Could not open webcam.")
    exit()

last_spoken_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("[Error] Failed to read frame.")
        break

    height, width = frame.shape[:2]

    # ── Detection ──────────────────────────────────────────────────────────
    boxes, confidences, class_ids, classes = detector.detect(frame)

    # ── Tracking ───────────────────────────────────────────────────────────
    detections = []
    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        detections.append(([x, y, w, h], confidences[i], class_ids[i]))

    tracks = tracker.update_tracks(detections, frame=frame)

    detected_now  = set()
    active_tracks = 0
    current_count = {}

    # ── Dashboard background ───────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 110), (20, 20, 20), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        class_id = track.det_class if track.det_class is not None else 0
        x1, y1, x2, y2 = map(int, track.to_ltrb())

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)

        class_name = classes[class_id] if class_id < len(classes) else "unknown"
        color      = get_color(class_id)
        box_w      = x2 - x1
        distance   = estimate_distance(box_w)

        detected_now.add(class_name)
        active_tracks += 1
        current_count[class_name] = current_count.get(class_name, 0) + 1

        # ── Trail ──────────────────────────────────────────────────────────
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append((cx, cy))
        if len(track_history[track_id]) > 30:
            track_history[track_id].pop(0)

        pts = track_history[track_id]
        for j in range(1, len(pts)):
            alpha = j / len(pts)
            thickness = max(1, int(3 * alpha))
            faded_color = tuple(int(c * alpha) for c in color)
            cv2.line(frame, pts[j - 1], pts[j], faded_color, thickness)

        # ── Bounding box ───────────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # ── Label ──────────────────────────────────────────────────────────
        label = f"#{track_id} {class_name} {distance}m" if distance else f"#{track_id} {class_name}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1 - 8, lh + 4)
        cv2.rectangle(frame, (x1, label_y - lh - 6), (x1 + lw + 4, label_y + 2), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 2, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # ── Dashboard text ─────────────────────────────────────────────────────
    cv2.putText(frame, "OBJECT DETECTION", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)
    cv2.putText(frame, f"Active Tracks : {active_tracks}", (10, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Detected      : {', '.join(detected_now) if detected_now else 'none'}", (10, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, f"FPS           : {int(cap.get(cv2.CAP_PROP_FPS))}", (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # ── Speech ─────────────────────────────────────────────────────────────
    if detected_now and (time.time() - last_spoken_time > SPEAK_INTERVAL):
        speech_text = ", ".join(
            f"{current_count[obj]} {obj}" if current_count[obj] > 1 else obj
            for obj in detected_now
        )
        speak(speech_text)
        last_spoken_time = time.time()

    # ── Show ───────────────────────────────────────────────────────────────
    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()