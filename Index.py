# ============================
#  🐟 FISH SPECIES PREDICTION (STREAMLIT)
# ============================

import streamlit as st
import os
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ============================
# 1️⃣ PAGE CONFIGURATION
# ============================

st.set_page_config(page_title="Fish Image Classification", layout="wide")
st.title("🐟 Multiclass Fish Image Classification")
st.text("")

# ============================
# 2️⃣ CREATE/LOAD CLASS LABELS JSON
# ============================

dataset_train_dir = r'D:\Fish_Detection\Dataset\images.cv_jzk6llhf18tm3k0kyttxz\data\train'
json_path = r'D:\Fish_Detection\Dataset\class_labels.json'

if not os.path.exists(json_path):
    st.warning("⚠️ class_labels.json not found — creating automatically...")
    classes = sorted(os.listdir(dataset_train_dir))
    labels = {str(i): cls for i, cls in enumerate(classes)}

    with open(json_path, 'w') as f:
        json.dump(labels, f, indent=4)
    st.success(f"✅ class_labels.json created at: {json_path}")
else:
    st.info("✅ class_labels.json found.")

# Load the labels
with open(json_path, 'r') as f:
    labels = json.load(f)

if isinstance(labels, dict):
    labels = [labels[str(i)] for i in range(len(labels))]

# ============================
# 3️⃣ MODEL SELECTION DROPDOWN
# ============================

model_options = {
    "Custom CNN": r"D:\Fish_Detection\MainApp\Custom CNN\custom_cnn_model.h5",
    "VGG16": r"D:\Fish_Detection\MainApp\VGG16\vgg_finetuned_model.h5",
    "ResNet50": r"D:\Fish_Detection\MainApp\ResNet50\resnet_finetuned_model.h5",
    "MobileNet": r"D:\Fish_Detection\MainApp\MobileNet\mobilenet_finetuned_model.h5",
    "InceptionV3": r"D:\Fish_Detection\MainApp\InceptionV3\inception_finetuned_model.h5",
    "EfficientNetB0": r"D:\Fish_Detection\MainApp\EfficientNetB0\efficientnet_finetuned_model.h5"
}

selected_model = st.selectbox("Select the model for prediction:", list(model_options.keys()))

# ============================
# 4️⃣ IMAGE UPLOAD & PREDICTION
# ============================

input_file = st.file_uploader("📁 Upload your fish image (JPG/PNG)", type=["jpg", "png"])

if input_file is not None:
    image = Image.open(input_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Classify"):
        model_path = model_options[selected_model]

        if os.path.exists(model_path):
            st.write(f"✅ Using model: **{selected_model}**")

            # Preprocess image
            img = image.convert("RGB").resize((224, 224))
            arr = np.array(img).astype("float32") / 255.0
            arr = np.expand_dims(arr, axis=0)

            # Load model and predict
            model = load_model(model_path)
            preds = model.predict(arr)
            probs = preds[0]

            # Top-3 predictions
            top_k = 3
            top_idx = probs.argsort()[-top_k:][::-1]

            st.subheader("🎯 Prediction Results")
            st.write(f"**Top-1:** 🐠 {labels[top_idx[0]]} — {probs[top_idx[0]]:.2%}")

            st.subheader("📊 Top-3 Probabilities")
            for i in top_idx:
                st.write(f"{labels[i]}: {probs[i]:.2%}")
        else:
            st.error(f"❌ Model not found at: {model_path}")

else:
    st.info("📤 Please upload an image to begin prediction.")
