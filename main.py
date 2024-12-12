import cv2
import pyaudio
import numpy as np
import time
from confluent_kafka import Producer
import json

# Initialize Kafka producer
producer = Producer({'bootstrap.servers': 'localhost:9092'})

def send_event(event_data):
    """Serialize event_data as a JSON string and send it to Kafka."""
    message = json.dumps(event_data)
    producer.produce('start_conversation', key='event', value=message)
    producer.flush()
    print(f"Sent event: {message}")  # Optional: Print the sent message for debugging

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Initialize audio capture
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100,
                    input=True, frames_per_buffer=1024)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# State variables
face_detected = False
audio_detected = False
last_cue_type = "none"  # Tracks the last sent cue type
last_event_time = 0  # Tracks the time of the last sent event
debounce_time = 5  # Minimum interval (in seconds) between events

try:
    while True:
        # Capture video frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from webcam.")
            break

        # Convert video frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Update face detection state
        face_detected = len(faces) > 0

        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Capture audio chunk
        try:
            audio_data = np.frombuffer(stream.read(
                1024, exception_on_overflow=False), dtype=np.int16)
            audio_level = np.max(audio_data)
        except Exception as e:
            print(f"Audio read error: {e}")
            audio_level = 0

        # Update audio detection state
        audio_detected = audio_level > 3000  # Adjust threshold as needed

        # Determine the current cue type
        if face_detected and audio_detected:
            current_cue_type = "both"
        elif face_detected:
            current_cue_type = "nonverbal"
        elif audio_detected:
            current_cue_type = "verbal"
        else:
            current_cue_type = "none"

        # Send an event only if the cue type changes and debounce interval has passed
        current_time = time.time()
        if current_cue_type != last_cue_type and (current_time - last_event_time) >= debounce_time:
            event_data = {
                "start_conversation": current_cue_type != "none",
                "cue_type": current_cue_type,
            }
            send_event(event_data)
            last_cue_type = current_cue_type  # Update the last cue type
            last_event_time = current_time  # Update the last event time

        # Display overlays based on detections
        if face_detected:
            cv2.putText(frame, "Face Detected!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if audio_detected:
            cv2.putText(frame, "Loud Audio!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the video feed with overlays
        cv2.imshow("Engagement Detection", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    stream.stop_stream()
    stream.close()
    audio.terminate()
