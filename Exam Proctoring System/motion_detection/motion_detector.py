import cv2
import time
import winsound  # For sound notification
import requests  # For sending request to trigger the alert
from deepface import DeepFace  # For emotion detection (requires installation)
import numpy as np  # For object detection

class MotionDetector:
    def __init__(self, motion_threshold=5, reset_interval=5):
        """
        Initialize the Motion Detector with motion and object detection capabilities.
        :param motion_threshold: Maximum allowed movements before triggering the alarm.
        :param reset_interval: Time interval in seconds to reset the motion count.
        """
        self.cap = cv2.VideoCapture(0)
        self.prev_frame = None
        self.motion_count = 0
        self.motion_threshold = motion_threshold
        self.reset_interval = reset_interval
        self.last_reset_time = time.time()
        self.alarm_triggered = False
        self.object_classes = None  # Store YOLO class labels

        # Load YOLO model for object detection
        self.net = cv2.dnn.readNet('yolov4.weights', 'yolov4.cfg')  # Use YOLOv4 weights and cfg
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        self.load_class_labels()

    def load_class_labels(self):
        """Load the class labels YOLO can detect."""
        with open('coco.names', 'r') as f:
            self.object_classes = f.read().strip().split('\n')

    def detect_motion(self, frame):
        # Preprocess the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray_frame
            return frame, False

        # Calculate frame differences
        diff_frame = cv2.absdiff(self.prev_frame, gray_frame)
        self.prev_frame = gray_frame

        # Threshold and find contours
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) < 5000:
                continue
            motion_detected = True
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if motion_detected:
            self.update_motion_count()

        return frame, motion_detected

    def detect_emotion(self, frame):
        """Detect emotion using DeepFace."""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            return emotion
        except Exception as e:
            print(f"Error in emotion detection: {e}")
            return None

    def detect_objects(self, frame):
        """Detect objects using YOLO."""
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Non-Maximum Suppression
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(self.object_classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame

    def update_motion_count(self):
        """Update motion count and trigger alarm if threshold exceeded."""
        current_time = time.time()
        
        # Reset motion count if interval has passed
        if current_time - self.last_reset_time > self.reset_interval:
            self.motion_count = 0
            self.alarm_triggered = False  # Reset alarm trigger flag
            self.last_reset_time = current_time

        # Increment motion count
        self.motion_count += 1

        # Trigger alarm if motion count exceeds threshold and alarm hasn't been triggered
        if self.motion_count > self.motion_threshold and not self.alarm_triggered:
            self.trigger_alarm()
            self.trigger_web_alert()  # Trigger the alert on the web interface
            self.alarm_triggered = True  # Set the flag to prevent multiple triggers

    def trigger_alarm(self):
        """Plays a beep sound when motion count exceeds the threshold."""
        winsound.Beep(1000, 1000)  # Frequency: 1000 Hz, Duration: 1000 ms (1 second)

    def trigger_web_alert(self):
        """Send request to trigger alert on web dashboard."""
        try:
            # Make a request to trigger the alert in the dashboard
            requests.get('http://localhost:5000/trigger_alert')
        except Exception as e:
            print(f"Error triggering web alert: {e}")

    def save_snapshot(self, frame):
        """Save snapshot when motion is detected."""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = f'static/snapshots/snapshot_{timestamp}.jpg'
        cv2.imwrite(filepath, frame)

    def get_video_stream(self):
        """Continuously capture video stream."""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, motion_detected = self.detect_motion(frame)
            if motion_detected:
                self.save_snapshot(frame)

            emotion = self.detect_emotion(frame)
            if emotion:
                cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            frame = self.detect_objects(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def release(self):
        """Release the video capture."""
        self.cap.release()
