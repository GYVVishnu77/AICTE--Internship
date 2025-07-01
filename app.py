import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model('EfficientV2B2_eWaste.keras')

class_names = [
    'Battery', 'Keyboard', 'Microwave', 'Mobile',
    'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine'
]

st.set_page_config(
    page_title="♻️ E-Waste Classifier",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ E-Waste Image Classifier")
st.markdown(
    "Upload an image of an e-waste item (like a mobile phone, PCB, or battery). "
    "Our AI model will predict its category to help sort it for recycling."
)

uploaded_file = st.file_uploader(
    "Choose an e-waste image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match model input size
    img = image.resize((260, 260))  # EfficientNetV2B2 input size
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    # Show result
    st.success(f"✅ **Prediction:** {class_names[index]}")
    st.info(f"📊 **Confidence:** {confidence:.2%}")

    # Show Top 3 predictions as a bar chart
    st.subheader("Top 3 Predictions")
    top3_indices = np.argsort(prediction[0])[::-1][:3]
    top3_labels = [class_names[i] for i in top3_indices]
    top3_scores = [prediction[0][i] for i in top3_indices]
    st.bar_chart({
        "Confidence": top3_scores
    })

st.markdown("---")
st.caption("🔬 Made with Streamlit & TensorFlow | 🌍 E-Waste Classification Project | 📧 Your Email Here")