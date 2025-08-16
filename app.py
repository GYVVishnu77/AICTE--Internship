import os
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# -----------------------------
# 0) CONFIG
# -----------------------------
H5_FILE_ID = "1rLPmVSrSlMCp8xr6-e4BHO0X8X-Zmvb8"   # Google Drive File ID of Efficient_classify.h5
H5_LOCAL   = "Efficient_classify_fixed.h5"

CLASS_NAMES = [
    "Battery", "Keyboard", "Microwave", "Mobile",
    "Mouse", "PCB", "Player", "Printer", "Television", "Washing Machine"
]

st.set_page_config(page_title="♻️ E-Waste Classifier", page_icon="♻️", layout="centered")
st.title("♻️ E-Waste Image Classifier")
st.markdown(
    "Upload an image of an e-waste item (like a **mobile phone**, **PCB**, or **battery**). "
    "Our AI model will predict its category to help sort it for recycling."
)

# -----------------------------
# 1) DOWNLOAD MODEL
# -----------------------------
def ensure_file(file_id: str, out_path: str):
    if file_id and (not os.path.exists(out_path)):
        gdown.download(id=file_id, output=out_path, quiet=False)
    return os.path.exists(out_path)

have_model = ensure_file(H5_FILE_ID, H5_LOCAL)

# -----------------------------
# 2) LOAD MODEL
# -----------------------------
model = None
load_error = None

if have_model:
    try:
        model = tf.keras.models.load_model(H5_LOCAL, compile=False)  # Direct load
    except Exception as e:
        load_error = e
        model = None

if model is None:
    st.error("❌ Could not load the model. Please check that your Google Drive file is public and the file ID is correct.")
    if load_error:
        st.exception(load_error)
    st.stop()

# -----------------------------
# 3) PREPROCESS HELPERS
# -----------------------------
TARGET_H, TARGET_W = 260, 260  # You trained with (260,260,3)

def preprocess(pil_img: Image.Image):
    img = pil_img.resize((TARGET_W, TARGET_H))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize
    arr = np.expand_dims(arr, axis=0)
    return arr

# -----------------------------
# 4) UPLOADER + PREDICTION
# -----------------------------
uploaded = st.file_uploader("📤 Upload an e-waste image (jpg/png)…", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    arr = preprocess(image)
    preds = model.predict(arr, verbose=0)[0]

    top_idx = int(np.argmax(preds))
    conf = float(preds[top_idx])

    st.success(f"✅ **Prediction:** {CLASS_NAMES[top_idx]}")
    st.info(f"📊 **Confidence:** {conf:.2%}")

    # Top-3 breakdown
    st.subheader("🔎 Top-3 predictions")
    order = np.argsort(preds)[::-1][:3]
    for i in order:
        st.write(f"- **{CLASS_NAMES[i]}** → {preds[i]:.2%}")

    st.progress(conf)

st.markdown("---")
st.caption("🔬 Built with **Streamlit** & **TensorFlow** · 🌍 E-Waste Classification Project · ✉️ your-email@example.com")

