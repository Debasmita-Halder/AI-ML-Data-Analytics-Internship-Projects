import cv2
from deepface import DeepFace
from collections import deque
import numpy as np

# Initialize a queue to store emotion probabilities for smoothing
emotion_queue = deque(maxlen=10)  # Store probabilities for the last 10 frames

def detect_emotion(frame):
    """
    Detects the emotion of a given video frame using DeepFace.
    Args:
        frame (ndarray): The video frame to analyze.
    Returns:
        str: The stabilized detected emotion, or 'neutral' if detection fails.
    """
    try:
        # Convert the frame to RGB (DeepFace requires RGB format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze the frame for emotions
        analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
        
        # Debugging: Print the raw analysis output
        print("Analysis Output:", analysis)
        
        # Extract emotions based on the output format
        if isinstance(analysis, list):
            emotions = analysis[0]["emotion"]
        elif isinstance(analysis, dict):
            emotions = analysis["emotion"]
        else:
            raise ValueError("Unexpected DeepFace output format")

        # Add current probabilities to the queue
        emotion_queue.append(emotions)

        # Average the probabilities over the queue for stabilization
        avg_emotions = {key: np.mean([e[key] for e in emotion_queue]) for key in emotions.keys()}

        # Determine the dominant emotion from averaged probabilities
        detected_emotion = max(avg_emotions, key=avg_emotions.get)
        return detected_emotion
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "neutral"

if __name__ == "__main__":
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    print("Press 'q' to quit the webcam.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Detect emotion
        emotion = detect_emotion(frame)
        print("Detected Emotion:", emotion)

        # Display the frame with stabilized detected emotion
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Emotion Detection", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
