import streamlit as st
from ultralytics import YOLO
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="SmartVision AI",
    page_icon="👁️",
    layout="wide"
)

st.title("SmartVision AI")
st.caption("End-to-End Object Detection → Classification Pipeline")

# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

yolo_model = load_yolo()

# ---------------------------------------------------
# LOAD CLASSIFIER (EfficientNet-B0)
# ---------------------------------------------------
NUM_CLASSES = 25

@st.cache_resource
def load_classifier():
    model = models.efficientnet_b0(weights=None)
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

# ---------------------------------------------------
# CLASSIFIER TRANSFORM
# ---------------------------------------------------
classifier_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
st.sidebar.header("⚙️ Detection Settings")

conf_threshold = st.sidebar.slider(
    "YOLO Confidence Threshold",
    0.1, 0.9, 0.25, 0.05
)

# ---------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("YOLOv8 Detection")

        results = yolo_model.predict(image, conf=conf_threshold)
        result = results[0]

        plotted = result.plot()
        st.image(plotted, use_container_width=True)

    # ---------------- CLASSIFICATION ----------------
    st.subheader("Detected Object Classification")

    if result.boxes is None or len(result.boxes) == 0:
        st.info("No objects detected.")
    else:
        boxes = result.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image.crop((x1, y1, x2, y2))

            input_tensor = classifier_transform(crop).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = classifier(input_tensor)
                probs = torch.softmax(outputs, dim=1)
                pred_class = torch.argmax(probs, dim=1).item() + 1  # start from 1
                confidence = probs[0][pred_class - 1].item()

            st.image(
                crop,
                caption=f"Predicted Class: {pred_class} | Confidence: {confidence:.2f}",
                width=200
            )

# ---------------------------------------------------
# MODEL PERFORMANCE TABLE (CLASSIFICATION ONLY)
# ---------------------------------------------------
st.markdown("---")
st.subheader("📊 Classification Model Performance")

comparison_data = {
    "Model": [
        "VGG16",
        "ResNet50",
        "MobileNetV2",
        "EfficientNet-B0"
    ],
    "Validation Accuracy (%)": [
        "82.3",
        "89.2",
        "63.2",
        "64.7"
    ]
}

df = pd.DataFrame(comparison_data)
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------
# YOLO METRIC (SEPARATE LINE)
# ---------------------------------------------------
st.markdown("""
**Detection Model (YOLOv8):**  
Evaluated using **mAP@50 (Mean Average Precision at IoU = 0.50)** → **0.129**

Unlike classification models that use accuracy, object detection models are
evaluated using localization-based metrics such as mAP.
""")

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------
st.markdown("---")
st.caption("SmartVision AI • End-to-End Computer Vision System")