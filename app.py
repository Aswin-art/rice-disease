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
SCALER_PATH = "scaler_padi.joblib"
ENCODER_PATH = "label_encoder_padi.joblib"

CLASS_TIPS = {
    "BrownSpot": (
        "Daun menunjukkan bercak-bercak cokelat kecil yang menyebar. "
        "Biasanya disebabkan oleh jamur. Periksa pola titik dan kelembapan area sawah."
    ),
    "LeafBlast": (
        "Terdapat bercak besar berbentuk belah ketupat atau memanjang pada daun. "
        "Sering muncul saat kondisi lembap dan pupuk nitrogen berlebih."
    ),
    "Hispa": (
        "Daun terlihat bergaris atau terkikis seperti goresan akibat serangan serangga. "
        "Biasanya terjadi karena kumbang Hispa memakan jaringan daun."
    ),
    "Healthy": (
        "Daun terlihat hijau segar dan merata tanpa bercak atau lesi yang mencolok. "
        "Tanaman berada dalam kondisi sehat."
    )
}


# ============================================================
# Load Artifacts
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    data = np.load(NPZ_PATH, allow_pickle=True)
    labels_map = data["labels_map"]
    return model, scaler, encoder, labels_map

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
def predict_image(file_bytes, model, scaler, encoder) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[np.ndarray], Optional[str]]:
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None, "Gambar tidak valid atau format tidak didukung."

    # Extract raw histogram features (512)
    raw_feat = extract_color_histogram_bgr(bgr).reshape(1, -1)
    # Scale features to match model's expected input dimensions
    feat = scaler.transform(raw_feat)
    # Predict class index
    pred_idx = model.predict(feat)[0]
    # Decode label using the saved encoder
    label = encoder.inverse_transform([pred_idx])[0]

    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(feat)[0]
        # Map probabilities to class names using encoder classes
        proba = {encoder.classes_[i]: float(p[i]) for i in range(len(encoder.classes_))}
        proba = dict(sorted(proba.items(), key=lambda x: x[1], reverse=True))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return label, proba, rgb, None

# ============================================================
# Helper
# ============================================================
def build_conclusion(label: str, proba: Optional[Dict[str, float]]) -> str:
    if not proba:
        base = f"🌱 **Kesimpulan:** Model memprediksi daun padi ini termasuk kategori **{label}** berdasarkan pola warna yang dianalisis."
        tip = CLASS_TIPS.get(label, "")
        return base + (f"\n\nℹ️ **Info tambahan:** {tip}" if tip else "")

    items = list(proba.items())
    top_label, top_conf = items[0][0], items[0][1]
    second = items[1] if len(items) > 1 else None

    if top_conf >= 0.80:
        conf_text = "tinggi"
        advice = "Model cukup yakin dengan hasil ini. Kamu bisa menjadikannya referensi utama."
    elif top_conf >= 0.60:
        conf_text = "menengah"
        advice = (
            "Model cukup yakin, tetapi akan lebih baik kalau kamu cek ulang pola bercak/lesi "
            "atau ambil foto lain dengan pencahayaan lebih merata."
        )
    else:
        conf_text = "rendah"
        advice = (
            "Model kurang yakin. Coba unggah foto lain dengan pencahayaan alami, fokus pada daun, "
            "atau lakukan pemeriksaan manual agar lebih pasti."
        )

    tip = CLASS_TIPS.get(top_label, "")

    lines = []
    lines.append(
        f"🌾 **Kesimpulan:** Berdasarkan analisis warna, daun ini **kemungkinan besar termasuk kategori `{top_label}`** "
        f"dengan tingkat keyakinan **{top_conf:.0%} ({conf_text})**."
    )

    if second:
        lines.append(
            f"🔎 Model juga melihat kemungkinan lain yaitu **`{second[0]}`** "
            f"dengan peluang sekitar **{second[1]:.0%}**."
        )

    if tip:
        lines.append(f"ℹ️ **Tentang `{top_label}`:** {tip}")

    lines.append(
        f"💡 **Saran:** {advice} "
        "Pastikan foto diambil dengan pencahayaan yang baik, tanpa bayangan tajam, "
        "dan daun memenuhi area gambar untuk hasil prediksi yang lebih akurat."
    )

    return "\n\n".join(lines)


# ============================================================
# UI
# ============================================================
st.title("🌾 Deteksi Penyakit Daun Padi (Random Forest + Histogram)")
st.write("Upload gambar daun padi lalu klik **Prediksi** untuk melihat hasil klasifikasi dan **kesimpulan** analisa.")

model, scaler, encoder, labels_map = load_artifacts()

if uploaded is not None:
    if st.button("Prediksi"):
        label, proba, rgb, err = predict_image(uploaded.read(), model, scaler, encoder)
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
