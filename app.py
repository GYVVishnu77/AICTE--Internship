import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# =============================
# 1. Download Model
# =============================
FILE_ID = "119PEoRPVNpwGYUWaBZEahkwrsyWhwfYH"  # Replace with your .keras file's ID
OUTPUT = "Efficient_classify.keras"

if not os.path.exists(OUTPUT):
    gdown.download(id=FILE_ID, output=OUTPUT, quiet=False)

# =============================
# 2. Load Trained Model
# =============================
model = tf.keras.models.load_model(OUTPUT)

# =============================
# 3. Class Names
# =============================
class_names = [
    "Battery", "Keyboard", "Microwave", "Mobile",
    "Mouse", "PCB", "Player", "Printer", "Television", "Washing Machine"
]

# =============================
# 4. Streamlit Page Setup
# =============================
st.set_page_config(
    page_title="♻️ E-Waste Classifier",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ E-Waste Image Classifier")
st.markdown(
    """
    Upload an image of an e-waste item (like a **mobile phone**, **PCB**, or **battery**).  
    Our AI model will automatically **predict its category** to help sort it for recycling.  
    """
)

# =============================
# 5. File Uploader
# =============================
uploaded_file = st.file_uploader(
    "📤 Upload an e-waste image...",
    type=["jpg", "jpeg", "png"]
)

# =============================
# 6. Prediction Logic
# =============================
if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model (resize to 260x260 since you trained on that)
    img = image.resize((260, 260))
    img_array = np.array(img, dtype=np.float32) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    # Show result
    st.success(f"✅ **Prediction:** {class_names[index]}")
    st.info(f"📊 **Confidence:** {confidence:.2%}")

    # =============================
    # 7. Top-3 Predictions
    # =============================
    st.subheader("🔎 Top 3 Predictions")
    top3_indices = np.argsort(prediction[0])[::-1][:3]
    top3_labels = [class_names[i] for i in top3_indices]
    top3_scores = [prediction[0][i] for i in top3_indices]

    # Display top-3 nicely
    for label, score in zip(top3_labels, top3_scores):
        st.write(f"- **{label}** → {score:.2%}")

    st.progress(float(confidence))  # fun confidence bar

# =============================
# 8. Footer
# =============================
st.markdown("---")
st.caption("🔬 Built with **Streamlit** & **TensorFlow** | 🌍 E-Waste Classification Project | ✉️ your-email@example.com")
