import sys
import os
import cv2

# Add the `src` directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from video_processing import detect_eye_contact, detect_head_nod

def main():
    """
    Main function for detecting eye contact and head nods using Mediapipe.
    """
    # Initialize variables
    prev_y_positions = []  # To track head nod movement history
    cap = cv2.VideoCapture(0)  # Open webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Detect eye contact
        eye_contact = detect_eye_contact(frame)

        # Detect head nods
        head_movement, prev_y_positions = detect_head_nod(frame, prev_y_positions)

        # Display results
        if eye_contact == "eye_contact":
            cv2.putText(frame, "Eye Contact Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if head_movement == "head_nod":
            cv2.putText(frame, "Head Nod Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the video feed with the detections
        cv2.imshow("Engagement Detection", frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")

if __name__ == "__main__":
    main()
