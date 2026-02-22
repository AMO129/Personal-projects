import streamlit as st
import cv2
import os
import time
from ultralytics import YOLO
import numpy as np

# --- 1. TACTICAL HUD THEMING (CSS) ---
st.set_page_config(page_title="Tactical Eye Command", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
        color: #00FF41; /* Tactical Matrix Green */
    }
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1A1C24;
        border-right: 2px solid #00FF41;
    }
    /* Critical Alert Flashing */
    @keyframes blinker {
        50% { opacity: 0; }
    }
    .critical-alert {
        color: #FF0000;
        font-weight: bold;
        animation: blinker 1s linear infinite;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. INFRASTRUCTURE ---
if not os.path.exists('captures'):
    os.makedirs('captures')


@st.cache_resource
def load_brain():
    return YOLO('yolov8n.pt')


model = load_brain()

# Session State for persistence
if 'last_saved_time' not in st.session_state:
    st.session_state.last_saved_time = 0
if 'target_confirmation' not in st.session_state:
    st.session_state.target_confirmation = {}

# --- 3. COMMAND SIDEBAR ---
st.sidebar.title("📡 HQ COMMAND")
st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider("Sensor Sensitivity", 0.0, 1.0, 0.65)
active = st.sidebar.toggle("Activate Defense Perimeter", value=True)

status_box = st.sidebar.empty()
log_box = st.sidebar.expander("Mission Logs", expanded=True)

# --- 4. TACTICAL FEED ---
st.write("### 📹 LIVE SECTOR SCAN")
frame_placeholder = st.empty()

cap = cv2.VideoCapture(0)
CONFIRMATION_THRESHOLD = 5

while cap.isOpened() and active:
    success, frame = cap.read()
    if not success: break

    # AI Inference
    results = model(frame, conf=conf_threshold)
    annotated_frame = results[0].plot()

    # Process detections with Tactical Logic
    for result in results:
        for box in result.boxes:
            label = model.names[int(box.cls[0])]
            conf = float(box.conf[0])

            st.session_state.target_confirmation[label] = st.session_state.target_confirmation.get(label, 0) + 1

            if st.session_state.target_confirmation[label] >= CONFIRMATION_THRESHOLD:
                if conf > 0.75:
                    status_box.markdown('<p class="critical-alert">⚠️ CRITICAL: TARGET CONFIRMED</p>',
                                        unsafe_allow_html=True)

                    # Evidence Capture
                    current_time = time.time()
                    if current_time - st.session_state.last_saved_time > 2:
                        timestamp = time.strftime("%H%M%S")
                        filename = f"captures/{label}_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        log_box.write(f"[{time.strftime('%H:%M')}] {label.upper()} SECURED")
                        st.session_state.last_saved_time = current_time
                else:
                    status_box.success("Perimeter Secure")

    # Display conversion (BGR to RGB)
    display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(display_frame, channels="RGB", use_container_width=True)

cap.release()
st.sidebar.info("System Standby.")