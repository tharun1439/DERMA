import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DermScan Pro", layout="wide")

st.markdown("""
    <style>
    body {background-color: #f4f9ff;}
    .big-font {font-size:22px !important;}
    .severe-alert {
        color: white;
        background-color: red;
        padding: 15px;
        border-radius: 10px;
        animation: blink 1s linear infinite;
        text-align: center;
        font-weight: bold;
    }
    @keyframes blink {
        50% {opacity: 0;}
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 DermScan Pro - Advanced Wound Infection Monitor")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Patient Details")
patient_name = st.sidebar.text_input("Enter Patient Name")

if patient_name == "":
    st.warning("Please enter patient name.")
    st.stop()

# ---------------- IMAGE SOURCE ----------------
input_method = st.radio(
    "Choose Image Source:",
    ["📷 Camera", "📤 Upload Image"]
)

img = None

# Camera
if input_method == "📷 Camera":
    camera_image = st.camera_input("Capture Image")
    if camera_image is not None:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

# Upload
elif input_method == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

# ---------------- PROCESSING ----------------
if img is not None:

    st.subheader("🔍 Processing Image...")

    # Resize for consistency
    img = cv2.resize(img, (400, 400))

    # Enhance contrast (CLAHE)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

    # 🔴 Red detection
    lower_red1 = np.array([0,120,70])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,120,70])
    upper_red2 = np.array([180,255,255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # 🟡 Yellow detection (pus)
    lower_yellow = np.array([20,100,100])
    upper_yellow = np.array([35,255,255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 📈 Texture detection
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Pixel calculations
    total_pixels = img.shape[0] * img.shape[1]
    red_pixels = cv2.countNonZero(red_mask)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    edge_pixels = cv2.countNonZero(edges)

    red_score = (red_pixels / total_pixels) * 100
    yellow_score = (yellow_pixels / total_pixels) * 100
    texture_score = (edge_pixels / total_pixels) * 100

    severity = (0.5 * red_score) + (0.3 * yellow_score) + (0.2 * texture_score)

    # ---------------- CONTOUR DETECTION ----------------
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = enhanced.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0, 0, 255), 3)

    # ---------------- GLOW OVERLAY ----------------
    overlay = enhanced.copy()
    overlay[red_mask > 0] = [0, 0, 255]
    blended = cv2.addWeighted(enhanced, 0.7, overlay, 0.3, 0)

    # ---------------- IMAGE DISPLAY ----------------
    st.markdown("## 🖼 Image Comparison")

    colA, colB = st.columns(2)
    colA.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original")
    colB.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), caption="Enhanced")

    st.markdown("### 🔬 Feature Maps")

    col1, col2, col3 = st.columns(3)
    col1.image(red_mask, caption="Red Detection")
    col2.image(yellow_mask, caption="Pus Detection")
    col3.image(edges, caption="Texture Map")

    st.markdown("### 🚨 Infection Highlight")
    st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), caption="Highlighted Infection Region")

    st.markdown("### 📦 Bounding Region")
    st.image(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB), caption="Detected Infected Area")

    # ---------------- METRICS ----------------
    st.subheader("📊 Infection Analysis")

    st.metric("Overall Severity (%)", f"{severity:.2f}")
    st.progress(int(min(severity, 100)))

    if severity < 15:
        st.success("🟢 Healing Normally")
        level = "Normal"
    elif severity < 35:
        st.warning("🟡 Moderate Infection Risk")
        level = "Moderate"
    else:
        st.markdown('<div class="severe-alert">🔴 SEVERE INFECTION! CONSULT DOCTOR IMMEDIATELY!</div>', unsafe_allow_html=True)
        level = "Severe"

    # ---------------- SAVE HISTORY ----------------
    data = {
        "Patient": patient_name,
        "Date": datetime.now(),
        "Severity": severity,
        "Level": level
    }

    df_new = pd.DataFrame([data])

    if os.path.exists("patients.csv"):
        df_old = pd.read_csv("patients.csv")
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_csv("patients.csv", index=False)

    # ---------------- PROGRESS GRAPH ----------------
    st.subheader("📈 Infection Trend")
    patient_data = df[df["Patient"] == patient_name]
    patient_data["Date"] = pd.to_datetime(patient_data["Date"])
    st.line_chart(patient_data.set_index("Date")["Severity"])

    # ---------------- DOWNLOAD REPORT ----------------
    report = f"""
    DermScan Medical Report
    ------------------------
    Patient Name: {patient_name}
    Date: {datetime.now()}
    Severity Score: {severity:.2f}%
    Infection Level: {level}
    """

    st.download_button(
        label="📥 Download Medical Report",
        data=report,
        file_name="DermScan_Report.txt",
        mime="text/plain"
    )