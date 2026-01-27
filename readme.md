# Smart Vision – Object Detection with YOLOv8

Smart Vision is an object detection project built using **YOLOv8** and **Streamlit** to demonstrate real-world capabilities and limitations of deep learning–based computer vision systems.

The project focuses on training a custom YOLOv8 detector on a **reduced set of 12 object classes**, highlighting how dataset size, class imbalance, and label quality affect detection performance.

---

## 🚀 Features
- Custom-trained **YOLOv8** object detection model
- Streamlit web app for image-based inference
- End-to-end ML pipeline:
  - Dataset preparation
  - Label remapping
  - Model training
  - Model evaluation
  - Deployment-ready inference UI

---

## 🧠 Model Details
- **Architecture:** YOLOv8n
- **Classes:** 12  
  (`person, car, bus, train, bicycle, motorcycle, airplane, dog, cat, cow, horse, zebra`)
- **Training Images:** ~125
- **Epochs:** 40
- **Hardware:** CPU-only training

---

## 📊 Results (mAP@50)
Overall mAP@50 ≈ **0.58**

Some classes (e.g. car, bus, train) perform well, while visually similar or underrepresented classes (e.g. zebra vs cat, airplane) show confusion — demonstrating **real-world model limitations**.

---

## ⚠️ Known Limitations
- Small dataset size
- Class imbalance
- Visually similar classes cause misclassification
- CPU-only training limits convergence

These limitations are **intentionally documented** as part of the learning objective.

---

## 🧩 Tech Stack
- Python
- Ultralytics YOLOv8
- PyTorch
- Streamlit
- OpenCV
- Git & GitHub

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
