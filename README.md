# Tactical Eye: AI-Driven Environmental Awareness System

## Overview
Tactical Eye is a real-time computer vision system designed for autonomous surveillance and threat detection. Built with **Python** and **YOLOv8**, the system monitors environments to identify and confirm tactical targets such as personnel and vehicles.

## Key Features
* **Temporal Confirmation:** Uses a frame-buffer logic to ensure targets are verified before alerting, reducing false positives.
* **Layered Threat Matrix:** Categorizes detections into 'SECURE' or 'CRITICAL' based on AI confidence thresholds.
* **Automated Evidence Capture:** High-resolution snapshots are secured automatically upon critical target confirmation.
* **Mission Logging:** Real-time console reports of environmental audits with timestamps.

## Technical Stack
* **Language:** Python
* **AI Model:** YOLOv8 (Ultralytics)
* **Vision Library:** OpenCV
* **Hardware Interface:** Standard CMOS Sensors (Webcam)

## How It Works (The Logic)
The system operates on a constant loop analyzing frames at a set confidence interval. By implementing a **Confirmation Threshold**, the system mimics human situational awareness by requiring a target to be present for a specific duration before initiating a response protocol.

---
*Developed by Moses Owalama Abraham.*