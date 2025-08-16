import os
import io
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown

# -----------------------------
# 0) CONFIG
# -----------------------------
KERAS_FILE_ID = "1hbCL5g_4HUUQe-jJVCTKV0pFLFLUXwHS"  # <-- YOUR .keras file ID
KERAS_LOCAL   = "eff_eWaste_model.keras"

# Optional fallback weights (only used if .keras fails to load)
# Put a valid Drive ID here if you also export weights with model.save_weights('...h5').
WEIGHTS_FILE_ID = None    # e.g. "1AbCxyz..." or keep None
WEIGHTS_LOCAL   = "eff_eWaste_weights.h5"

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
# 1) DOWNLOAD ARTIFACTS
# -----------------------------
def ensure_file(file_id: str, out_path: str):
    if file_id and (not os.path.exists(out_path)):
        gdown.download(id=file_id, output=out_path, quiet=False)
    return os.path.exists(out_path)

have_keras = ensure_file(KERAS_FILE_ID, KERAS_LOCAL)
have_weights = ensure_file(WEIGHTS_FILE_ID, WEIGHTS_LOCAL) if WEIGHTS_FILE_ID else False

# -----------------------------
# 2) LOAD / REBUILD MODEL
# -----------------------------
model = None
load_error = None

if have_keras:
    try:
        # Keras 3 defaults to safe_mode=True; set False to allow Lambda/legacy graphs if present.
        model = tf.keras.models.load_model(KERAS_LOCAL, compile=False, safe_mode=False)
    except Exception as e:
        load_error = e
        model = None

if (model is None) and have_weights:
    # Fallback: rebuild the exact head you trained and load weights.
    # You trained at 260x260; if you changed that, update here.
    try:
        base = tf.keras.applications.EfficientNetV2B2(
            input_shape=(260, 260, 3), include_top=False, weights=None
        )
        x = tf.keras.layers.Input(shape=(260, 260, 3))
        y = base(x, training=False)
        y = tf.keras.layers.GlobalAveragePooling2D()(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        out = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax")(y)
        model = tf.keras.Model(x, out)
        model.load_weights(WEIGHTS_LOCAL)
        load_error = None
    except Exception as e:
        load_error = e
        model = None

# Hard fail with a clear message if we didn’t get a model.
if model is None:
    st.error("❌ Could not load the model. Please check that your Google Drive file is public and the file ID is correct.")
    if load_error:
        st.exception(load_error)
    st.stop()

# -----------------------------
# 3) PREPROCESS HELPERS
# -----------------------------
def model_expects_size(m):
    """Return (H, W) expected by the model from its input shape."""
    ish = m.input_shape
    # input_shape may be a list (Functional w/ multiple inputs). Pick the first.
    if isinstance(ish, (list, tuple)) and isinstance(ish[0], (list, tuple)):
        ish = ish[0]
    # shape like (None, H, W, C)
    return int(ish[1]), int(ish[2])

def model_already_rescales(m):
    """Detect if a Rescaling layer (scale ~1/255) exists near the input."""
    for layer in m.layers[:5]:  # quick scan the early layers
        if isinstance(layer, tf.keras.layers.Rescaling):
            # common training: Rescaling(scale=1./255)
            return True
    return False

TARGET_H, TARGET_W = model_expects_size(model)
DO_RESCALE = not model_already_rescales(model)

st.caption(f"Model expects images of **{TARGET_W}×{TARGET_H}**. "
           f"Input rescaling in app: **{'yes' if DO_RESCALE else 'no (model handles it)'}**.")

# -----------------------------
# 4) UPLOADER + PREDICTION
# -----------------------------
uploaded = st.file_uploader("📤 Upload an e-waste image (jpg/png)…", type=["jpg", "jpeg", "png"])

def predict_image(pil_img: Image.Image):
    img = pil_img.resize((TARGET_W, TARGET_H))
    arr = np.asarray(img, dtype=np.float32)
    if DO_RESCALE:
        arr = arr / 255.0
    arr = np.expand_dims(arr, axis=0)
    # If you trained with EfficientNetV2 preprocessing and NO Rescaling layer,
    # you can uncomment the next two lines instead of manual 1/255:
    # from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    # arr = preprocess_input(arr)
    preds = model.predict(arr, verbose=0)[0]
    return preds

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    preds = predict_image(image)
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

