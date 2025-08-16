from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response
import cv2
import os
from motion_detection.motion_detector import MotionDetector
from datetime import datetime, timedelta
import random
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

# Dummy user storage (This can be replaced with a database in production)
users = {}

# Initialize the face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Motion Detector
motion_detector = MotionDetector(motion_threshold=5, reset_interval=5)

# Ensure snapshots directory exists
os.makedirs('static/snapshots', exist_ok=True)

# Home Route
@app.route('/')
def home():
    return render_template('home.html')

# Exam Route
@app.route('/exam')
def exam():
    return render_template('exam.html')



@app.route('/thankyou')
def thankyou():
    return render_template('thankyou.html')

# About Route
@app.route('/about')
def about():
    return render_template('about.html')

# Contact Route
@app.route('/contact')
def contact():
    return render_template('contact.html')

# Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add logic to check user credentials and set session
        session['username'] = username
        return redirect(url_for('home'))
    return render_template('login.html')

# Registration Route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        # Add logic to store user data (e.g., in a database)
        return redirect(url_for('login'))
    return render_template('register.html')

# Logout Route
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove user from session
    flash('You have been logged out', 'info')
    return redirect(url_for('home'))

@app.route('/motion-detected')
def motion_detected():
    # Run the motion detection method, if motion is detected return True
    detected = motion_detector.detect_motion()
    return jsonify({'motionDetected': detected})

@app.route('/video_feed')
def video_feed():
    return Response(motion_detector.get_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshots')
def snapshots():
    files = os.listdir('static/snapshots')
    files.sort(reverse=True)  # Show recent snapshots first
    return jsonify(files)
# Exam Route
@app.route('/question')
def question():
    return render_template('question.html')

# Route for face detection
# @app.route('/detect_face', methods=['POST'])
# def detect_face():
#     # Get the base64-encoded image from the POST request
#     data = request.get_json()
#     image_data = data['image']
    
#     # Decode the base64 image
#     img_data = base64.b64decode(image_data.split(',')[1])
#     image = Image.open(BytesIO(img_data))
#     image = np.array(image)

#     # Convert to grayscale for face detection
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect faces in the image
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
#     if len(faces) > 0:
#         # Return the coordinates of the first detected face (you can modify to return more if needed)
#         (x, y, w, h) = faces[0]
#         return jsonify({'left': x, 'top': y, 'width': w, 'height': h})
#     else:
#         return jsonify({'left': 0, 'top': 0, 'width': 0, 'height': 0})





if __name__ == '__main__':
    app.run(debug=True)
