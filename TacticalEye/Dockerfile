# Start with a lightweight Python foundation
FROM python:3.10-slim

# Set the working directory on the server
WORKDIR /app

# Install the critical C++ system libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy your requirements file and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your code into the server
COPY . .

# Open the port Streamlit uses
EXPOSE 8501

# The command to boot up the tactical feed
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]