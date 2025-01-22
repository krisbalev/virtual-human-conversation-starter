import json
import time
from confluent_kafka import Producer
from eye_nod_detection import EyeNodDetector
from voice_detection import VoiceDetector

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

    # Initialize Detectors
    detector = EyeNodDetector(
        camera_index=0,
        nod_threshold=15.0,
        cooldown=1.0,
        alpha=0.3,
        iris_center_threshold_ratio=0.25,
        max_yaw_for_gaze=50
    )
    voice_detector = VoiceDetector(model_path="vosk_model/vosk-model-en-us-0.22")

    # State Variables
    conversation_active = False
    eye_contact_prev_frame = False
    eye_contact_start_time = 0.0
    no_eye_contact_start_time = 0.0
    last_nod_time = 0.0

    print("Starting. Press ESC to stop...")

    try:
        while True:
            # Get cues from EyeNodDetector
            nod_detected, eye_contact, should_quit = detector.get_cues_and_show()

            # Get speech from VoiceDetector
            voice_command = voice_detector.detect_speech()

            current_time = time.time()

            if not conversation_active:
                if eye_contact:
                    # Start eye contact timer if it's the first frame
                    if not eye_contact_prev_frame:
                        eye_contact_start_time = current_time
                    
                    # Condition 1: Eye contact + nod + no speech for 3 seconds
                    if nod_detected and not voice_command:
                        last_nod_time = current_time
                    elif last_nod_time > 0 and (current_time - last_nod_time) >= 3.0:
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
                        print("Triggered conversation with nod and no speech for 3 seconds.")
                        conversation_active = True

                    # Condition 2: Eye contact + speech
                    elif voice_command:
                        payload = {
                            "start_conversation": True,
                            "type": "MULTIMODAL",
                            "cues": ["eye_contact", "voice_command"],
                            "spoken_text": voice_command
                        }
                        producer.produce(
                            topic=topic_name,
                            key="start_conversation",
                            value=json.dumps(payload),
                            callback=delivery_report
                        )
                        print(f"Triggered conversation with speech: {voice_command}")
                        conversation_active = True

                    # Condition 3: Eye contact sustained for 5 seconds
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
                    # Reset timers if no eye contact
                    eye_contact_start_time = 0.0
                    last_nod_time = 0.0

            else:
                # Handle conversation end
                if not eye_contact:
                    if eye_contact_prev_frame:
                        no_eye_contact_start_time = current_time
                    elif (current_time - no_eye_contact_start_time) >= 10.0:
                        print("Conversation ended due to loss of eye contact.")
                        conversation_active = False

            # Update previous frame
            eye_contact_prev_frame = eye_contact
            producer.poll(0)

            if should_quit:
                break

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nUser interrupted with Ctrl+C.")
    finally:
        detector.release()
        voice_detector.stop()
        producer.flush()
        print("Shutting down.")

if __name__ == "__main__":
    main()
