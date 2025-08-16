from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
from emotion_model import detect_emotion 
import random

app = Flask(__name__)

# Load emotion-to-song mapping
music_recommendations = {
        "happy": [
            {"title": "Supernatural", "artist": "New Jeans", "image": "static/nj2.jpg"},
            {"title": "Armageddon", "artist": "Aespa", "image": "static/aespa.webp"},
            {"title": "Snooze", "artist": "SZA", "image": "static/SZA.webp"},
            {"title": "Best Part", "artist": "Daniel Caeser", "image": "static/best part.jpeg"}
        ],
        "sad": [
            {"title": "Supernatural", "artist": "New Jeans", "image": "static/nj2.jpg"},
            {"title": "Armageddon", "artist": "Aespa", "image": "static/aespa.webp"},
            {"title": "Snooze", "artist": "SZA", "image": "static/SZA.webp"},
            {"title": "Best Part", "artist": "Daniel Caeser", "image": "static/best part.jpeg"}
        ],
        "angry": [
            {"title": "Supernatural", "artist": "New Jeans", "image": "static/nj2.jpg"},
            {"title": "Armageddon", "artist": "Aespa", "image": "static/aespa.webp"},
            {"title": "Snooze", "artist": "SZA", "image": "static/SZA.webp"},
            {"title": "Best Part", "artist": "Daniel Caeser", "image": "static/best part.jpeg"}
        ],
        "neutral": [
            {"title": "Supernatural", "artist": "New Jeans", "image": "static/nj2.jpg"},
            {"title": "Armageddon", "artist": "Aespa", "image": "static/aespa.webp"},
            {"title": "Snooze", "artist": "SZA", "image": "static/SZA.webp"},
            {"title": "Best Part", "artist": "Daniel Caeser", "image": "static/best part.jpeg"}
        ],
        "surprise": [
           {"title": "Supernatural", "artist": "New Jeans", "image": "static/nj2.jpg"},
            {"title": "Armageddon", "artist": "Aespa", "image": "static/aespa.webp"},
            {"title": "Snooze", "artist": "SZA", "image": "static/SZA.webp"},
            {"title": "Best Part", "artist": "Daniel Caeser", "image": "static/best part.jpeg"}
        ],
        "fear": [
            {"title": "Supernatural", "artist": "New Jeans", "image": "static/nj2.jpg"},
            {"title": "Armageddon", "artist": "Aespa", "image": "static/aespa.webp"},
            {"title": "Snooze", "artist": "SZA", "image": "static/SZA.webp"},
            {"title": "Best Part", "artist": "Daniel Caeser", "image": "static/best part.jpeg"}
        ]
    }
    


# Function to recommend a song based on emotion
def recommend_song(emotion, song_data):
    if emotion in song_data:
        return song_data[emotion]  # Return the whole list of songs
    else:
        return ["No recommendation available."]


cap = cv2.VideoCapture(0)
@app.route("/")
def home():
    return render_template("home.html")
        

@app.route('/r')
def r():
    # Capture emotion and recommend music
    _, frame = cap.read()
    detected_emotion = detect_emotion(frame)  # Use your existing emotion detection model
    
    # Fetch the music recommendations based on detected emotion
    recommendations = music_recommendations.get(detected_emotion, music_recommendations['neutral'])

    return render_template('r.html', emotion=detected_emotion, recommendations=recommendations)

@app.route('/video_feed')
def video_feed():
    # Function to stream the video feed
    _, frame = cap.read()
    ret, jpeg = cv2.imencode('.jpg', frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')
@app.route("/about")
def about():
    return render_template("about.html")
@app.route("/contact")
def contact():
    return render_template("contact.html")
               

if __name__ == "__main__":
    app.run(debug=True)
