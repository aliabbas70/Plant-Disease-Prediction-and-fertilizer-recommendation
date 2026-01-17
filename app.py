"""
app.py
AI Based Plant Disease Prediction & Fertilizer Recommendation

Models expected at:
 - ./2.keras
 - ./random_forest_classifier_model.pkl

Developed by: Ali Abbas Abdi | BCA Mini Project
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageOps
import io
import pandas as pd
from datetime import datetime
import os

# -----------------------
# Config / Constants
# -----------------------
DISEASE_MODEL_PATH = "2.keras"
FERTILIZER_MODEL_PATH = "random_forest_classifier_model.pkl"

IMAGE_SIZE = 256
CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Fertilizer info (edit according to your label classes)
FERTILIZER_INFO = {
    "Fertilizer_A": {
        "description": "High Nitrogen fertilizer for leafy growth.",
        "dosage": "Apply 25 kg per acre. For potted plant: 2 g per plant.",
        "safety": "Wear gloves. Keep away from children."
    },
    "Fertilizer_B": {
        "description": "Balanced NPK fertilizer for general purpose.",
        "dosage": "Apply 20 kg per acre. For potted plant: 1.5 g per plant.",
        "safety": "Store in dry place."
    },
    "Fertilizer_C": {
        "description": "High Potassium fertilizer to improve flowering/fruiting.",
        "dosage": "Apply 15 kg per acre. For potted plant: 2 g per plant.",
        "safety": "Avoid contact with eyes."
    }
}

# -----------------------
# Helpers
# -----------------------
@st.cache_resource
def load_models():
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
    with open(FERTILIZER_MODEL_PATH, "rb") as f:
        fertilizer_model = pickle.load(f)
    return disease_model, fertilizer_model

def preprocess_image(pil_img: Image.Image, size=IMAGE_SIZE):
    img = pil_img.convert("RGB")
    img = ImageOps.fit(img, (size, size))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def predict_disease(model, pil_img):
    x = preprocess_image(pil_img)
    preds = model.predict(x)
    probs = preds.flatten()
    idx = int(np.argmax(probs))
    class_name = CLASS_NAMES[idx]
    confidence = float(probs[idx])
    return class_name, confidence, probs

def get_time_now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_sample_image_paths():
    folder = "sample_images"
    if not os.path.isdir(folder):
        return []
    exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(exts)]

# -----------------------
# Load models
# -----------------------
with st.spinner("Loading AI models..."):
    disease_model, fertilizer_model = load_models()

# -----------------------
# Page config & style
# -----------------------
st.set_page_config(page_title="AI Based Plant Disease Prediction & Fertilizer Recommendation", layout="wide")

st.markdown(
    """
    <style>
    .app-title {text-align:center; color:#22543D; font-size:30px; font-weight:700;}
    .subtitle {text-align:center; color:#2F855A; margin-bottom:20px;}
    .footer {text-align:center; color: #718096; font-size:12px; margin-top:20px;}
    .card {background-color:#F7FAFC; padding:12px; border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------
# Top Banner
# -----------------------
st.markdown('<div class="app-title">🌾 AI Based Plant Disease Prediction & Fertilizer Recommendation</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf photo or enter soil N-P-K values to get AI suggestions. (Info Only)</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("Go to Plant Disease Prediction"):
        st.session_state["active_tab"] = "disease"
with col_b:
    if st.button("Go to Fertilizer Recommendation"):
        st.session_state["active_tab"] = "fert"

st.markdown("---")

# -----------------------
# TABS
# -----------------------
tabs = st.tabs([
    "🏠 Home",
    "🌱 Plant Disease Prediction",
    "🧪 Fertilizer Recommendation",
    "📜 History",
    "ℹ️ About / Help"
])

# -----------------------
# HOME
# -----------------------
with tabs[0]:
    st.markdown("### Quick Links")
    c1, c2, c3 = st.columns(3)
    c1.write("🌱 Plant Disease Prediction")
    c2.write("🧪 Fertilizer Recommendation")
    c3.write("📜 History & Export")

    st.markdown("---")
    st.markdown("### Sample Images")
    sample_paths = get_sample_image_paths()
    if sample_paths:
        cols = st.columns(min(4, len(sample_paths)))
        for i, p in enumerate(sample_paths[:4]):
            cols[i].image(Image.open(p), caption=os.path.basename(p))
    else:
        st.info("Create a folder named `sample_images/` and add images.")

    st.markdown("### How to Take a Good Image")
    st.write("""
    - Use daylight  
    - Focus leaf clearly  
    - Avoid shadows  
    - Use plain background  
    """)

# -----------------------
# DISEASE PREDICTION
# -----------------------
with tabs[1]:
    st.header("🌱 Upload Leaf Image")

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
        crop_selector = st.selectbox("Crop (optional)", ["Potato", "Apple", "Corn", "Other"])
        enable_crop = st.checkbox("Enable manual crop", False)

        crop_region = None
        if uploaded_file and enable_crop:
            img = Image.open(uploaded_file)
            w, h = img.size
            left = st.slider("Left", 0, w-1, 0)
            top = st.slider("Top", 0, h-1, 0)
            right = st.slider("Right", left+1, w, w)
            bottom = st.slider("Bottom", top+1, h, h)
            crop_region = (left, top, right, bottom)

        predict = st.button("Predict Disease")

    with col2:
        if uploaded_file:
            st.image(Image.open(uploaded_file), caption="Uploaded Image", use_column_width=True)
        else:
            st.info("Upload an image to preview here.")

    if predict:
        if not uploaded_file:
            st.error("Upload an image first!")
        else:
            img = Image.open(uploaded_file)
            if crop_region:
                img = img.crop(crop_region)

            with st.spinner("Predicting..."):
                pred_class, conf, probs = predict_disease(disease_model, img)

            st.success(f"Predicted: **{pred_class}**  |  Confidence: {conf*100:.2f}%")

            # interpretation
            if conf >= 0.85:
                st.write("Confidence: High")
            elif conf >= 0.60:
                st.write("Confidence: Moderate")
            else:
                st.write("Confidence: Low — try another clearer image.")

            st.markdown("### Steps (Informational Only)")
            if pred_class == "Potato___Early_blight":
                st.write("- Remove affected leaves\n- Apply mancozeb\n- Improve ventilation")
            elif pred_class == "Potato___Late_blight":
                st.write("- Remove infected plants\n- Apply systemic fungicide\n- Act quickly")
            elif pred_class == "Potato___healthy":
                st.write("- Plant is healthy\n- Continue monitoring")

            # probability table
            df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": [f"{float(p)*100:.2f}%" for p in probs]
            })
            st.table(df)

            # history save
            hs = st.session_state.setdefault("history", [])
            hs.insert(0, {
                "timestamp": get_time_now_str(),
                "type": "disease",
                "crop": crop_selector,
                "predicted_class": pred_class,
                "confidence": conf,
                "npk": None
            })

# -----------------------
# FERTILIZER RECOMMENDATION
# -----------------------
with tabs[2]:
    st.header("🧪 Enter N-P-K Values")

    col1, col2, col3 = st.columns(3)
    N = col1.number_input("Nitrogen (N)", 0.0, 500.0, 50.0)
    P = col2.number_input("Phosphorus (P)", 0.0, 500.0, 30.0)
    K = col3.number_input("Potassium (K)", 0.0, 500.0, 40.0)

    if st.button("Recommend Fertilizer"):
        with st.spinner("Analyzing..."):
            pred = fertilizer_model.predict([[N, P, K]])[0]

        st.success(f"Recommended: **{pred}**")
        if pred in FERTILIZER_INFO:
            info = FERTILIZER_INFO[pred]
            st.write("**Description:**", info["description"])
            st.write("**Dosage:**", info["dosage"])
            st.write("**Safety:**", info["safety"])
        else:
            st.info("Add fertilizer description in FERTILIZER_INFO.")

        # save history
        hs = st.session_state.setdefault("history", [])
        hs.insert(0, {
            "timestamp": get_time_now_str(),
            "type": "fertilizer",
            "crop": None,
            "predicted_class": pred,
            "confidence": None,
            "npk": {"N": N, "P": P, "K": K}
        })

# -----------------------
# HISTORY
# -----------------------
with tabs[3]:
    st.header("📜 History")
    hs = st.session_state.setdefault("history", [])

    if not hs:
        st.info("No history yet.")
    else:
        def norm(x):
            return {
                "timestamp": x["timestamp"],
                "type": x["type"],
                "crop": x.get("crop",""),
                "result": x["predicted_class"],
                "confidence": f"{x['confidence']:.3f}" if x["confidence"] is not None else "",
                "N": x["npk"]["N"] if x["npk"] else "",
                "P": x["npk"]["P"] if x["npk"] else "",
                "K": x["npk"]["K"] if x["npk"] else ""
            }

        df = pd.DataFrame([norm(x) for x in hs])
        st.dataframe(df, use_container_width=True)

        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")

        if st.button("Clear History"):
            st.session_state["history"] = []
            st.rerun()

# -----------------------
# ABOUT
# -----------------------
with tabs[4]:
    st.header("ℹ️ About / Help")
    st.write("""
    - CNN Model for Potato Disease Classification  
    - RandomForest Model for Fertilizer Recommendation  
    - For Mac M1: install tensorflow-macos + tensorflow-metal  
    """)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="footer">© Ali Abbas Abdi | BCA Mini Project</div>', unsafe_allow_html=True)
