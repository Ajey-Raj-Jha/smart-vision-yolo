import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Smart Vision – Object Detection",
    layout="centered"
)

st.title("🧠 Smart Vision – Object Detection Demo")
st.write(
    "Upload an image and detect objects using a YOLOv8 model trained on a reduced 12-class dataset."
)

# Load model (cache so it doesn't reload every time)
@st.cache_resource
def load_model():
    model_path = r"C:/Users/ajeyr/runs/detect/train16/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run inference
    st.subheader("Detection Result")
    with st.spinner("Running detection..."):
        results = model.predict(
            source=temp_path,
            conf=0.45,
            iou=0.5,
            save=False
        )

    # Show result
    result_img = results[0].plot()
    st.image(result_img, use_column_width=True)

    # Cleanup
    os.remove(temp_path)

    st.success("Detection completed!")

st.markdown("---")
st.caption(
    "⚠️ Note: Misclassifications may occur due to limited dataset size and visual similarity between classes."
)
