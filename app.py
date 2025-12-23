import streamlit as st
import numpy as np
import cv2
import joblib
from typing import Dict, Tuple, Optional
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew

# ============================================================
# App Config
# ============================================================
st.set_page_config(page_title="Deteksi Penyakit Daun Padi", page_icon="🌾")

MODEL_PATH = "random_forest_model.joblib"
SCALER_PATH = "scaler_padi.joblib"
ENCODER_PATH = "label_encoder_padi.joblib"
IMG_SIZE = (256, 256)

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
    return model, scaler, encoder

# ============================================================
# Feature Extraction (sama dengan training)
# ============================================================
def extract_features_v2(img):
    """Extract 43 features: HSV color stats + GLCM texture + LBP histogram"""
    img = cv2.resize(img, IMG_SIZE)

    # A. Segmen HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 20, 20])
    upper = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # B. Warna (9 fitur: 3 channels × 3 stats)
    color_feats = []
    for i in range(3):
        res = hsv[:,:,i][mask > 0]
        if len(res) > 0:
            color_feats.extend([np.mean(res), np.std(res), skew(res)])
        else:
            color_feats.extend([0, 0, 0])

    # C. Tekstur GLCM (8 fitur: 4 properties × 2 angles)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi/4], levels=256, symmetric=True, normed=True)
    for p in ['contrast', 'homogeneity', 'energy', 'correlation']:
        color_feats.extend(graycoprops(glcm, p).flatten())

    # D. Tekstur LBP (26 fitur: 26 bins histogram)
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return np.hstack([color_feats, hist])

# ============================================================
# Inference
# ============================================================
def predict_image(file_bytes, model, scaler, encoder) -> Tuple[Optional[str], Optional[Dict[str, float]], Optional[np.ndarray], Optional[str]]:
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None, None, None, "Gambar tidak valid atau format tidak didukung."

    # Extract 43 features (HSV + GLCM + LBP)
    raw_feat = extract_features_v2(bgr).reshape(1, -1)
    # Scale features
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
        f"🌾 **Kesimpulan:** Berdasarkan analisis fitur warna dan tekstur, daun ini **kemungkinan besar termasuk kategori `{top_label}`** "
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
st.title("🌾 Deteksi Penyakit Daun Padi (Random Forest)")
st.write("Upload gambar daun padi lalu klik **Prediksi** untuk melihat hasil klasifikasi dan **kesimpulan** analisa.")

uploaded = st.file_uploader("Upload gambar (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])
model, scaler, encoder = load_artifacts()

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
