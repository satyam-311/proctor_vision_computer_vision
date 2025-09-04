# ProctorVision without MediaPipe
# Works on Python 3.11

import cv2
import time
from collections import deque

# Optional YOLO for phone detection
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ---------- Config ----------
CAM_INDEX = 0
CONF_THRESHOLD = 0.5
SMOOTH_WINDOW = 5
MULTI_FACE_FRAMES = 5
ABSENT_FRAMES = 15
RUN_PHONE_DET = True

# ---------- Helper functions ----------
def moving_avg(queue, val, maxlen):
    queue.append(val)
    if len(queue) > maxlen:
        queue.popleft()
    return sum(queue) / len(queue)

def draw_flag(frame, text, y, color=(0, 0, 255)):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

# ---------- Main ----------
def main():
    phone_model = None
    if RUN_PHONE_DET and YOLO_AVAILABLE:
        phone_model = YOLO(r"C:\Users\Satyam Mishra\.ipython\yolov8n.pt")  # use your downloaded YOLO model
        print("YOLO model loaded. Classes:", phone_model.names)

    # Load Haar cascade from your local path
    face_cascade = cv2.CascadeClassifier(r"C:\Users\Satyam Mishra\Downloads\haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade XML. Check the path!")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    absent_counter = 0
    multi_face_counter = 0
    face_x_hist = deque()
    frame_counter = 0
    last_log = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        face_count = len(faces)

        # ---------- Face presence / multiple faces ----------
        if face_count == 0:
            absent_counter += 1
        else:
            absent_counter = 0

        if face_count > 1:
            multi_face_counter += 1
        else:
            multi_face_counter = 0

        # ---------- Approximate "looking away" ----------
        looking_away = False
        if face_count >= 1:
            x, y, w, h = faces[0]
            cx = x + w/2
            cx_smooth = moving_avg(face_x_hist, cx, SMOOTH_WINDOW)

            if cx_smooth < frame.shape[1]*0.25 or cx_smooth > frame.shape[1]*0.75:
                looking_away = True

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # ---------- Phone detection (every 5 frames) ----------
        phone_seen = False
        frame_counter += 1
        if RUN_PHONE_DET and phone_model is not None:
            if frame_counter % 5 == 0:  # run YOLO every 5 frames
                results_yolo = phone_model.predict(source=frame, verbose=False, imgsz=640, conf=CONF_THRESHOLD)
                for r in results_yolo:
                    for box, cls, conf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
                        name = r.names[int(cls)]
                        print(f"Detected: {name} ({conf:.2f})")  # debug
                        if name == "cell phone" and float(conf) >= CONF_THRESHOLD:
                            phone_seen = True
                            x1, y1, x2, y2 = map(int, box.tolist())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                            cv2.putText(frame, f"PHONE {conf:.2f}", (x1, y1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # ---------- Flags / Alerts ----------
        y_cursor = 24
        if absent_counter >= ABSENT_FRAMES:
            draw_flag(frame, "ALERT: No face detected", y_cursor); y_cursor += 24
        if multi_face_counter >= MULTI_FACE_FRAMES:
            draw_flag(frame, "ALERT: Multiple faces detected", y_cursor); y_cursor += 24
        if face_count >= 1 and looking_away:
            draw_flag(frame, "ALERT: Looking away / off-screen", y_cursor); y_cursor += 24
        if RUN_PHONE_DET and phone_seen:
            draw_flag(frame, "ALERT: Phone detected", y_cursor); y_cursor += 24

        # ---------- Console log ----------
        now = time.time()
        if now - last_log > 1.0:
            last_log = now
            print({
                "faces": face_count,
                "absent_counter": absent_counter,
                "multi_face_counter": multi_face_counter,
                "looking_away": looking_away,
                "phone": phone_seen
            })

        cv2.imshow("ProctorVision (Press Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
