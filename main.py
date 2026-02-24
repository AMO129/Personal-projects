import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# --- 1. TACTICAL HUD THEMING (CSS) ---
st.set_page_config(page_title="Tactical Eye Command", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #00FF41; 
    }
    section[data-testid="stSidebar"] {
        background-color: #1A1C24;
        border-right: 2px solid #00FF41;
    }
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


# --- 2. INFRASTRUCTURE & AI CORE ---
@st.cache_resource
def load_brain():
    # Caching prevents reloading the model every time a user connects
    return YOLO('yolov8n.pt')


model = load_brain()

# WebRTC requires STUN servers to navigate firewalls on public networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- 3. COMMAND SIDEBAR ---
st.sidebar.title("📡 HQ COMMAND")
st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider("Sensor Sensitivity", 0.0, 1.0, 0.65)

# --- 4. TACTICAL FEED (WEBRTC CALLBACK) ---
st.write("### 📹 LIVE SECTOR SCAN")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """
    This function processes the incoming video frames from the user's browser.
    """
    # Convert WebRTC frame to an OpenCV-compatible array
    img = frame.to_ndarray(format="bgr24")

    # AI Inference
    results = model(img, conf=conf_threshold)

    # Plot the bounding boxes on the frame
    annotated_frame = results[0].plot()

    # Convert the processed array back to a WebRTC frame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


# Initialize the WebRTC Streamer
webrtc_streamer(
    key="tactical-radar",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.sidebar.info("System Standby. Grant camera permissions in browser to initialize perimeter scan.")