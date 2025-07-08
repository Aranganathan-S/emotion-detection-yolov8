# Real-time Emotion Detection using YOLOv8 🎯

This project performs real-time facial emotion detection using a custom-trained YOLOv8 model and OpenCV. It captures webcam feed, detects faces, and classifies them into one of eight emotions using bounding boxes, emotion labels, and confidence scores.

---

## 💡 Motivation

Built with the vision of integrating into urban CCTV networks to detect emotional distress and alert authorities in real time.

---

## 🧠 Model Overview

- **Model:** YOLOv8s (Ultralytics)
- **Dataset:** 9,400 labeled facial expression images (via Roboflow)
- **Classes:** `anger`, `content`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
- **Weights:** `best.pt` (22MB) — included in this repo

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- Roboflow (for dataset export)

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Aranganathan-S/emotion-detection-yolov8.git
cd emotion-detection-yolov8

# 2. Install dependencies
pip install ultralytics opencv-python

# 3. Ensure best.pt is in the repo folder (already included if you cloned the full repo)

# 4. Run the emotion detection script
python realtime_emotion_detector.py
```

This opens your webcam and starts detecting emotions live with bounding boxes, confidence scores, and FPS.

---

## 📁 Folder Structure

```
emotion-detection-yolov8/
├── best.pt                      # YOLOv8 trained model (22MB)
├── realtime_emotion_detector.py # Main script
├── README.md                    # This file
├── .gitignore                   # Ignore .pt, __pycache__, etc.
```

---

## 📺 Output

- Real-time webcam feed
- Detected faces with colored bounding boxes
- Emotion labels with confidence scores
- FPS counter in top-left

---

## 📌 Notes

- Modify `cv2.VideoCapture(0)` in the script to run on CCTV or video files.
- This model is lightweight and runs smoothly on most modern machines.
- `best.pt` is included here since it's under GitHub's 100MB file limit.

---

## ✍️ Author

**Aranganathan S**  
🔗 [LinkedIn](https://www.linkedin.com/in/aranganathan-s) • 🤗 [Hugging Face](https://huggingface.co/Aranganathan-S)
