# main.py

import json
import time
from confluent_kafka import Producer

from eye_nod_detection import EyeNodDetector

def main():
    # ------------- Confluent Kafka Producer Setup -------------
    conf = {
        'bootstrap.servers': 'localhost:9092',  # or your Docker host:port
    }
    producer = Producer(conf)

    def delivery_report(err, msg):
        """Called once for each message produced to indicate delivery result."""
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            # Decode the message value from bytes to string (assuming UTF-8)
            payload = msg.value().decode('utf-8')
            # Decode the key if present
            key = msg.key().decode('utf-8') if msg.key() else "None"
            print(f"Message delivered to {msg.topic()} [{msg.partition()}] "
                  f"- key: {key}, value: {payload}")

    topic_name = "start_conversation"

    # ------------- Initialize the Detector -------------
    detector = EyeNodDetector(
        camera_index=0,       # change if you have multiple cameras
        nod_threshold=15.0,
        cooldown=1.0,
        alpha=0.3,
        iris_center_threshold_ratio=0.25,
        max_yaw_for_gaze=50
    )

    # ------------- State Variables -------------
    conversation_active = False

    eye_contact_session_start = 0.0
    no_eye_contact_start = 0.0

    # Track whether the user had eye contact on the previous frame
    eye_contact_prev_frame = False

    print("Starting. Press ESC to stop...")

    try:
        while True:
            nod_detected, eye_contact, should_quit = detector.get_cues_and_show()

            if not conversation_active:
                # We're currently not in a conversation

                if eye_contact:
                    # 1) Eye contact + nod => instant start
                    if nod_detected:
                        payload = {
                            "start_conversation": True,
                            "type": "NONVERBAL",
                            "cues": ["eye_contact", "head_nod"]  # both cues
                        }
                        producer.produce(
                            topic=topic_name,
                            key="start_conversation",
                            value=json.dumps(payload),
                            callback=delivery_report
                        )
                        conversation_active = True

                    else:
                        # 2) Only eye contact => wait 5 seconds
                        if not eye_contact_prev_frame:
                            # Just started eye contact, record time
                            eye_contact_session_start = time.time()
                        else:
                            # Already in eye contact
                            if (time.time() - eye_contact_session_start) >= 5.0:
                                # Start conversation after 5s
                                payload = {
                                    "start_conversation": True,
                                    "type": "NONVERBAL",
                                    "cues": ["eye_contact"]  # only eye contact
                                }
                                producer.produce(
                                    topic=topic_name,
                                    key="start_conversation",
                                    value=json.dumps(payload),
                                    callback=delivery_report
                                )
                                conversation_active = True
                else:
                    # No eye contact => reset the timer
                    eye_contact_session_start = 0.0

            else:
                # conversation_active == True
                # Ignore nods entirely

                if eye_contact:
                    # If user re-establishes eye contact before 10 seconds, do nothing
                    pass
                else:
                    # No eye contact this frame
                    # If we JUST broke eye contact, record the time
                    if eye_contact_prev_frame:
                        no_eye_contact_start = time.time()
                    else:
                        # We've been out of eye contact for some time
                        # check if it's >= 10 seconds
                        if (time.time() - no_eye_contact_start) >= 10.0:
                            print("conversation ended")
                            conversation_active = False
                            # If you also want to send a Kafka message for conversation ended, do:
                            # payload = {
                            #     "conversation_ended": True,
                            #     "timestamp": time.time()
                            # }
                            # producer.produce(
                            #     topic=topic_name,
                            #     key="conversation_ended",
                            #     value=json.dumps(payload),
                            #     callback=delivery_report
                            # )

            # Update previous eye contact
            eye_contact_prev_frame = eye_contact

            # Let the producer handle events
            producer.poll(0)

            # Check for ESC
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
