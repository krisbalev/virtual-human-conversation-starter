import json
import time
from confluent_kafka import Producer
from eye_nod_detection import EyeNodDetector

def main():
    # Kafka Producer Setup
    conf = {
        'bootstrap.servers': 'localhost:9092',
    }
    producer = Producer(conf)

    def delivery_report(err, msg):
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            payload = msg.value().decode('utf-8')
            key = msg.key().decode('utf-8') if msg.key() else "None"
            print(f"Message delivered to {msg.topic()} [{msg.partition()}] - key: {key}, value: {payload}")

    topic_name = "start_conversation"

    # --------------------------------------------------------------------
    # Initialize the EyeNodDetector with parameters you can adjust.
    # --------------------------------------------------------------------
    detector = EyeNodDetector(
        camera_index=0,
        nod_pitch_diff_threshold=15.0,  # how big a pitch change is considered a nod
        min_nod_frames=3,              # frames for nod pattern (inside EyeNodDetector)
        nod_cooldown=1.0,              # min seconds between nod detections
        eye_contact_frames_required=5,  # frames needed to confirm eye contact internally
        iris_center_threshold_ratio=0.25,
        max_yaw_for_gaze=50,           
        alpha=0.3,                     
        pitch_buffer_size=10           
    )

    # --------------------------------------------------------------------
    # Conversation state and timers
    # --------------------------------------------------------------------
    conversation_active = False

    eye_contact_prev_frame = False
    eye_contact_start_time = 0.0       # For 5-second eye contact tracking
    no_eye_contact_start_time = 0.0    # For 10-second eye contact loss
    print("Starting. Press ESC to stop...")

    try:
        while True:
            # 1) Detect nod/eye_contact from the EyeNodDetector
            nod_detected, eye_contact, should_quit = detector.get_cues_and_show()
            current_time = time.time()

            # ------------------ START Conversation Logic ------------------
            if not conversation_active:
                if eye_contact:
                    # If we just established eye contact, record its start time
                    if not eye_contact_prev_frame:
                        eye_contact_start_time = current_time

                    # Condition A: Immediate start if eye_contact + nod_detected
                    if nod_detected:
                        payload = {
                            "start_conversation": True,
                            "type": "NONVERBAL",
                            "cues": ["eye_contact", "head_nod"]
                        }
                        producer.produce(
                            topic=topic_name,
                            key="start_conversation",
                            value=json.dumps(payload),
                            callback=delivery_report
                        )
                        print("Triggered conversation immediately on eye contact + nod.")
                        conversation_active = True

                    # Condition B: Eye contact for 5 consecutive seconds
                    elif (current_time - eye_contact_start_time) >= 5.0:
                        payload = {
                            "start_conversation": True,
                            "type": "NONVERBAL",
                            "cues": ["eye_contact"]
                        }
                        producer.produce(
                            topic=topic_name,
                            key="start_conversation",
                            value=json.dumps(payload),
                            callback=delivery_report
                        )
                        print("Triggered conversation with sustained eye contact for 5 seconds.")
                        conversation_active = True

                else:
                    # No eye contact => reset the eye_contact_start_time
                    eye_contact_start_time = 0.0

            # ------------------- END Conversation Logic -------------------
            else:
                # If conversation is active, check if we lose eye contact for 10 seconds
                if not eye_contact:
                    # If we just lost eye contact, note the time
                    if eye_contact_prev_frame:
                        no_eye_contact_start_time = current_time
                    else:
                        # If we've had no eye contact for 10 seconds, end
                        if (current_time - no_eye_contact_start_time) >= 10.0:
                            print("Conversation ended due to loss of eye contact.")
                            conversation_active = False

            # Update eye_contact status for the next iteration
            eye_contact_prev_frame = eye_contact

            # Process any outstanding Kafka events
            producer.poll(0)

            # If ESC was pressed, break the loop
            if should_quit:
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nUser interrupted with Ctrl+C.")
    finally:
        detector.release()
        producer.flush()
        print("Shutting down.")

if __name__ == "__main__":
    main()
