import json
from confluent_kafka import Consumer, KafkaException

# Initialize Kafka consumer
consumer_conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'engagement_group',
    'auto.offset.reset': 'latest'  # Start from the latest messages
}
consumer = Consumer(consumer_conf)
consumer.subscribe(['engagement_topic'])

def process_event(event_data):
    if event_data.get('cue_type') == 'both' and event_data.get('start_conversation'):
        # Both verbal and nonverbal cues are present
        respond_with_greeting()
    else:
        print("No greeting needed.")

def respond_with_greeting():
    greeting = "Hello! How can I assist you today?"
    print(f"Greeting sent: {greeting}")
    # Implement additional logic to send this greeting elsewhere if needed

try:
    while True:
        msg = consumer.poll(1.0)  # Timeout of 1 second
        if msg is None:
            continue  # No message received
        if msg.error():
            raise KafkaException(msg.error())
        else:
            # Message received
            raw_value = msg.value()
            if raw_value is None:
                print("Received message with empty value.")
                continue
            decoded_value = raw_value.decode('utf-8')
            print(f"Raw message value: '{decoded_value}'")
            try:
                event_data = json.loads(decoded_value)
                print(f"Event received: {event_data}")
                process_event(event_data)
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}")
except KeyboardInterrupt:
    print("Consumer interrupted by user.")
finally:
    # Close down consumer to commit final offsets.
    consumer.close()
