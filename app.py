import streamlit as st
import numpy as np
import cv2
import joblib
from typing import Dict, Tuple, Optional

# ============================================================
# App Config
# ============================================================
st.set_page_config(page_title="Deteksi Penyakit Daun Padi", page_icon="🌾")

MODEL_PATH = "random_forest_model.joblib"
NPZ_PATH   = "fitur_histogram.npz"

CLASS_TIPS = {
    "BrownSpot": "Bercak cokelat pada daun, periksa pola titik menyebar.",
    "LeafBlast": "Daun berbentuk belah ketupat; sering muncul memanjang pada daun.",
    "Hispa": "Kerusakan karena serangga, tampak goresan/strip dan skeletonisasi.",
    "Healthy": "Daun tampak hijau merata tanpa bercak/lesi mencolok."
}

# ============================================================
# Load Artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    labels_map = data["labels_map"]
    return model, labels_map

# ============================================================
# Feature Extraction
# ============================================================
def extract_color_histogram_bgr(bgr_image, bins=(8, 8, 8)):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([rgb], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# ============================================================
# Inference
# ============================================================
def predict_image(file_bytes, model, labels_map) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[np.ndarray], Optional[str]]:
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None, "Gambar tidak valid atau format tidak didukung."

    feat = extract_color_histogram_bgr(bgr).reshape(1, -1)
    pred_idx = model.predict(feat)[0]
    label = str(labels_map[pred_idx])

    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(feat)[0]
        proba = {str(labels_map[i]): float(p[i]) for i in range(len(labels_map))}
        proba = dict(sorted(proba.items(), key=lambda x: x[1], reverse=True))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return label, proba, rgb, None

# ============================================================
# Helper
# ============================================================
def build_conclusion(label: str, proba: Optional[Dict[str, float]]) -> str:
    if not proba:
        base = f"**Kesimpulan:** Model memprediksi kelas **{label}** berdasarkan analisis histogram warna daun."
        tip  = CLASS_TIPS.get(label, "")
        return base + (f"\n\n**Catatan:** {tip}" if tip else "")

    items = list(proba.items())
    top_label, top_conf = items[0][0], items[0][1]
    second = items[1] if len(items) > 1 else None

    if top_conf >= 0.80:
        conf_text = "tinggi"
        advice = "Hasil ini **cukup dapat diandalkan**."
    elif top_conf >= 0.60:
        conf_text = "menengah"
        advice = (
            "Sebaiknya **verifikasi manual** (cek pola bercak/lesi) "
            "atau ambil ulang foto dengan pencahayaan merata."
        )
    else:
        conf_text = "rendah"
        advice = (
            "Model **kurang yakin**. Disarankan unggah foto lain (pencahayaan alami, fokus pada daun) "
            "atau lakukan pemeriksaan lebih lanjut."
        )

    tip = CLASS_TIPS.get(top_label, "")

    lines = []
    lines.append(f"**Kesimpulan:** Citra **kemungkinan besar adalah `{top_label}`** "
                 f"dengan **kepercayaan {top_conf:.0%}** (keyakinan {conf_text}).")
    if second:
        lines.append(f"Prediksi alternatif: `{second[0]}` ({second[1]:.0%}).")

    if tip:
        lines.append(f"**Catatan `{top_label}`:** {tip}")

    lines.append(
        f"**Rekomendasi:** {advice} "
        "Hindari bayangan keras, pantulan kuat, dan pastikan daun memenuhi area gambar."
    )

    return "\n\n".join(lines)

# ============================================================
# UI
# ============================================================
st.title("🌾 Deteksi Penyakit Daun Padi (Random Forest + Histogram)")
st.write("Upload gambar daun padi lalu klik **Prediksi** untuk melihat hasil klasifikasi dan **kesimpulan** analisa.")

uploaded = st.file_uploader("Upload gambar (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

model, labels_map = load_artifacts()

if uploaded is not None:
    if st.button("Prediksi"):
        label, proba, rgb, err = predict_image(uploaded.read(), model, labels_map)
        if err:
            st.error(err)
        else:
            st.image(rgb, caption="Gambar diunggah", use_container_width=True)

            st.subheader("Hasil Prediksi")
            st.markdown(f"- **Kelas utama:** `{label}`")
            if proba:
                st.markdown("- **Probabilitas per kelas:**")
                st.bar_chart(proba)

            st.subheader("Kesimpulan")
            conclusion = build_conclusion(label, proba)
            st.markdown(conclusion)
