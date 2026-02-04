import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np


# ---------------- Page Config ----------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="👁️",
    layout="wide"
)

st.markdown(
    """
    <h1 style="text-align:center;">SmartVision AI</h1>
    <p style="text-align:center; color:gray;">
    End-to-End Object Detection Pipeline using YOLOv8
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- Load YOLO Model ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
# ---------------- Load EfficientNetB0 Classifier ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 25

@st.cache_resource
def load_classifier():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        NUM_CLASSES
    )
    model.load_state_dict(
        torch.load("efficientnetb0_final.pth", map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    return model

classifier = load_classifier()

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Settings")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.25,
    step=0.05
)

st.sidebar.markdown(
    """
    **Model:** YOLOv8  
    **Dataset:** COCO 25-class subset  
    **Pipeline:** Detection → Crop → (Classification)
    """
)

# ---------------- Helper: Crop detections ----------------
def crop_detections(image, result):
    crops = []
    boxes = result.boxes

    for box, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        x1, y1, x2, y2 = map(int, box)
        crop = image.crop((x1, y1, x2, y2))

        crops.append({
            "image": crop,
            "yolo_class": int(cls),
            "confidence": float(conf)
        })

    return crops

# ---------------- Main UI ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ---------- Original Image ----------
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_container_width=True)

    # ---------- YOLO Detection ----------
    with col2:
        st.subheader("🔍 YOLO Detection")

        with st.spinner("Running YOLOv8 inference..."):
            results = model.predict(image, conf=conf_threshold)
            result_img = results[0].plot()

        st.image(result_img, use_container_width=True)

# ---------------- Model Comparison Section ----------------
st.markdown("---")
st.subheader("📊 Model Comparison & Selection")

comparison_data = {
    "Model": ["VGG16", "ResNet50", "MobileNetV2", "EfficientNetB0"],
    "Parameters (M)": [134.36, 23.56, 2.26, 4.04],
    "Model Size (MB)": [512.57, 90.18, 8.84, 15.70],
    "Inference Time (ms)": [21.53, 9.04, 5.90, 11.46]
}

st.table(comparison_data)

st.markdown(
    """
**Analysis & Final Selection**

VGG16, while simple and effective, suffers from an extremely large parameter count and
high memory usage, making it unsuitable for deployment-focused applications.
MobileNetV2 achieves the fastest inference speed and smallest footprint, making it ideal
for edge and mobile environments, though with a slight accuracy trade-off.
ResNet50 provides a strong balance between accuracy and efficiency.
EfficientNetB0 delivers the best overall trade-off by combining high accuracy with a
compact model size and reasonable inference latency.

**EfficientNetB0 was therefore selected as the final classification model and integrated
with the YOLOv8 detection pipeline.**
"""
)

# ---------------- Footer ----------------
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:gray;">
    SmartVision AI • Phase 4.1 Pipeline Complete
    </p>
    """,
    unsafe_allow_html=True
)
