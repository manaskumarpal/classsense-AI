# ClassSense AI

An AI-powered classroom analytics prototype designed to assist teachers with classroom monitoring, attendance analysis, student analytics, and classroom event reporting.

## 🚀 Project Overview

ClassSense AI combines computer vision, machine learning, database management, and a web dashboard into one classroom analytics prototype.

The current version demonstrates how classroom camera data could be processed to generate useful insights for teachers.

## ✨ Features

- Teacher Login
- Classroom Dashboard
- YOLO-based object detection
- Person, phone and laptop detection
- Classroom session management
- Classroom event logging
- Machine Learning based student analytics
- Random Forest classification
- Attendance management
- Student analytics
- Session reports
- Event reports
- Student reports
- Attendance reports
- CSV report downloads
- SQLite database
- Streamlit web interface

## 🧠 AI Technologies

### Computer Vision

YOLO is used for object detection from classroom images and camera streams.

### Machine Learning

A Random Forest classifier is used to categorize student analytics into:

- Excellent
- Good
- Needs Attention

The current training dataset is synthetic and is intended for demonstrating the ML pipeline.

## 🏗️ System Architecture

```text
Classroom Camera / Image
          ↓
     YOLO Detection
          ↓
    Event Processing
          ↓
     ML Analytics
          ↓
       Database
          ↓
   Streamlit Dashboard
          ↓
 Teacher Reports & Insights

 🛠️ Technologies Used
Python
Streamlit
YOLO
Ultralytics
Scikit-learn
Random Forest
OpenCV
Pandas
NumPy
SQLite
Joblib
Pillow

Project structure
ClassSense-AI/
│
├── app.py
├── ml_model.py
├── train_model.py
├── database.py
├── attendance.py
├── requirements.txt
├── training_data.csv
├── classsense_model.pkl
├── yolo11n.pt
└── README.md

How to Run

1. Install dependencies
python -m pip install -r requirements.txt

2. Train the ML model
python train_model.py

3. Start the application
streamlit run app.py

The application will open in the browser.

🔐 Demo Login
Username: teacher
Password: classsense123

📊 Current Version

Version 1.0 — Functional Prototype

This version is a proof-of-concept demonstrating the proposed AI classroom analytics workflow.

The current ML demonstration uses synthetic data and has not been validated on a real classroom dataset.

🔮 Future Development

Future versions can include:

Continuous-presence based automatic attendance
Configurable attendance time threshold
Real classroom IP camera integration
Ceiling-mounted camera deployment
Edge AI processing
Real-world classroom dataset
Improved model validation
Multi-classroom support
Scalable cloud/database architecture
🎥 Proposed Real-World Deployment
Ceiling-Mounted IP Camera
            ↓
       Network / PoE
            ↓
       Edge AI Device
            ↓
     YOLO + Tracking
            ↓
 Attendance + Analytics
            ↓
         Database
            ↓
     Teacher Dashboard
⚠️ Important Note

ClassSense AI is a classroom-support prototype.

Its analytics are intended to provide indicators for teacher review and should not automatically determine student marks, promotion, or disciplinary decisions.

The prototype uses anonymous/synthetic student identifiers for demonstration.

👨‍💻 Project Status

Current prototype successfully demonstrates the core software workflow.

Future work will focus on real-world camera integration, continuous attendance tracking, real datasets, model validation, and deployment testing.