📚 Exam Doctoring System
🔎 Overview

The Exam Doctoring System is an AI-powered proctoring application designed to monitor online examinations and ensure academic integrity. It leverages computer vision and deep learning techniques to detect suspicious activities such as unauthorized movement, object detection, and emotional behavior analysis during exams.

This system integrates seamlessly into an online exam platform to provide real-time monitoring, alert generation, and reporting.

⚙️ Features

🎥 Motion Detection – Detects unusual or excessive movements by the candidate.

🧑‍🤝‍🧑 Face Detection – Ensures only the registered candidate is present.

🙂 Emotion Detection – Tracks candidate’s emotional state to identify stress or abnormal behavior.

🎒 Object Detection (YOLO) – Detects unauthorized objects like mobile phones, books, or other people in the frame.

🔔 Real-Time Alerts – Raises warnings if suspicious activities are detected.

📸 Snapshots & Logs – Captures frames as evidence when violations occur.

🌐 Web Integration – Runs within a Flask-based web application with live video streaming.

🛠️ Technologies Used

Programming Language: Python

Frameworks: Flask (for web integration), OpenCV (for computer vision)

AI/ML Models:

DeepFace (for emotion recognition)

Haar Cascade / DNN models (for face detection)

YOLO (You Only Look Once) for object detection

Frontend: HTML, CSS, JavaScript (Flask templates)

Backend: Flask with video streaming support

Others: NumPy, Pandas, Pickle (for model handling), Logging & Alerts

🚀 Workflow

User Login → Candidates log in securely before starting the exam.

Live Video Monitoring → Camera feed is analyzed in real time.

Detection Pipeline:

Motion detection tracks unusual activity.

Face detection ensures single-user presence.

Emotion detection monitors stress and anomalies.

Object detection identifies prohibited materials.

Alert Generation → Suspicious activity triggers alerts and snapshots.

Report Generation → At the end, logs and evidence are stored for examiner review.



📌 Notes

Pre-trained models (YOLO weights, DeepFace dependencies) may not be included due to size constraints.

Please download the required models from their official repositories and place them in the models/ folder.

A Google Drive link to models can also be provided here if needed.

✨ Future Improvements

Cloud deployment for large-scale online exams.

Multi-camera support for wider monitoring.

Advanced behavioral analytics with NLP for voice monitoring.
