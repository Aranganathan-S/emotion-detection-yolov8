import cv2
from ultralytics import YOLO
import threading
import queue
import torch

# Load model
model = YOLO("best.pt")

if torch.cuda.is_available():
    model.to("cuda")

frame_queue = queue.Queue(maxsize=5)
result_queue = queue.Queue(maxsize=5)

cap = cv2.VideoCapture(0)

color_map = {
    'anger': (0, 0, 255),
    'content': (255, 255, 0),
    'disgust': (0, 255, 0),
    'fear': (255, 0, 255),
    'happy': (0, 255, 255),
    'neutral': (128, 128, 128),
    'sad': (255, 0, 0),
    'surprise': (0, 165, 255)
}

# ---------------- Capture Thread ----------------
def capture_frames():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

# ---------------- Inference Thread ----------------
def inference():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            results = model.predict(frame, imgsz=640, conf=0.5, verbose=False)
            result_queue.put((frame, results))

# ---------------- Display Thread ----------------
def display():
    while True:
        if not result_queue.empty():
            frame, results = result_queue.get()

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    confidence = box.conf[0].item()
                    label = model.names[cls_id]
                    coords = box.xyxy[0].cpu().numpy().astype(int)

                    x1, y1, x2, y2 = coords
                    color = color_map.get(label, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}',
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Live Emotion Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Start threads
threading.Thread(target=capture_frames, daemon=True).start()
threading.Thread(target=inference, daemon=True).start()
threading.Thread(target=display, daemon=True).start()

# Keep main alive
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
