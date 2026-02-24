# Start with a lightweight Python foundation
FROM python:3.10-slim

# Set the working directory on the server
WORKDIR /app

# THE HEAVY ARMOR: Every single system library required by OpenCV and PyAV (WebRTC)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    pkg-config \
    python3-dev \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code into the server
COPY . .

# Open the port Streamlit uses
EXPOSE 8501

# The command to boot up the tactical feed
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]