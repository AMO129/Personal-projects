import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from twilio.rest import Client

# --- TACTICAL HUD THEMING ---
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


# --- INFRASTRUCTURE & AI CORE ---
@st.cache_resource
def load_brain():
    # Load the model. It will automatically download yolov8n.pt if missing.
    return YOLO('yolov8n.pt')


model = load_brain()

# ----FETCH TWILIO TURN SERVERS---
@st.cache_data 
def get_ice_servers():
    try:
          #Pull credentials safely from Streamlit Secrets 
            account_sid = st.secrets["TWILIO_ACCOUNT_SID"]
            auth_token = st.secrets["TWILIO_AUTH_TOKEN"]

            client = Client(account_sid, auth_token)
            token = client.tokens.create()
            return
            token.ice_servers
except Exception as e:

           st.warning("Twilio credentials not found, falling back to free STUN.")
           return [{"urls":["stun:stun.l.google.com:19302"]}]

#---RTC FUNCTION---
RTC_CONFIGURATION = RTCConfiguration({"iceServers":get_ice_servers()})


# --- COMMAND SIDEBAR ---
st.sidebar.title("📡 HQ COMMAND")
st.sidebar.markdown("---")
conf_threshold = st.sidebar.slider("Sensor Sensitivity", 0.0, 1.0, 0.65)

# --- TACTICAL FEED ---
st.write("### 📹 LIVE SECTOR SCAN")


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert WebRTC frame to an OpenCV array
    img = frame.to_ndarray(format="bgr24")

    # AI Inference (verbose=False prevents log spam on the server)
    results = model(img, conf=conf_threshold, verbose=False)

    # Plot bounding boxes
    annotated_frame = results[0].plot()

    # Convert back to WebRTC frame
    return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


webrtc_streamer(
    key="tactical-radar",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.sidebar.info("System Standby. Grant camera permissions in browser to initialize perimeter scan.")
