import streamlit as st
import numpy as np
import cv2
import joblib
from io import BytesIO

MODEL_PATH = 'random_forest_model.joblib'
NPZ_PATH   = 'fitur_histogram.npz'

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    labels_map = data['labels_map']
    return model, labels_map

def extract_color_histogram_bgr(bgr_image, bins=(8, 8, 8)):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([rgb], [0,1,2], None, bins, [0,256, 0,256, 0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def predict(image_bytes, model, labels_map):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        return "Gambar tidak valid", None, None

    feat = extract_color_histogram_bgr(bgr).reshape(1, -1)
    pred_idx = model.predict(feat)[0]
    class_name = str(labels_map[pred_idx])

    proba_dict = None
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(feat)[0]
        proba_dict = {str(labels_map[i]): float(prob[i]) for i in range(len(labels_map))}
        proba_dict = dict(sorted(proba_dict.items(), key=lambda x: x[1], reverse=True))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return class_name, proba_dict, rgb

st.set_page_config(page_title="Deteksi Penyakit Daun Padi", page_icon="🌾", layout="centered")
st.title("🌾 Deteksi Penyakit Daun Padi (Random Forest + Histogram)")
st.write("Upload gambar daun padi, lalu klik **Prediksi**.")

uploaded = st.file_uploader("Pilih file gambar", type=["jpg","jpeg","png"])

model, labels_map = load_artifacts()

if uploaded is not None:
    if st.button("Prediksi"):
        label, proba, rgb = predict(uploaded, model, labels_map)
        if rgb is not None:
            st.image(rgb, caption="Gambar diupload", use_column_width=True)
        st.subheader(f"Prediksi: **{label}**")
        if proba:
            st.bar_chart(proba)
