from confluent_kafka import Consumer, KafkaException, KafkaError

def main():
    # Kafka Consumer configuration
    conf = {
        'bootstrap.servers': 'localhost:9092',  # or your Docker host:port
        'group.id': 'test-consumer-group',      # Consumer group id
        'auto.offset.reset': 'earliest',        # Start reading at the earliest message
    }

    consumer = Consumer(conf)
    topic_name = "start_conversation"

    # Subscribe to the topic
    consumer.subscribe([topic_name])

    print(f"Listening for messages on topic '{topic_name}'... Press Ctrl+C to stop.")

    try:
        while True:
            msg = consumer.poll(1.0)  # Poll for a message, with a timeout of 1 second
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition event
                    print(f"Reached end of partition: {msg.error()}")
                elif msg.error():
                    raise KafkaException(msg.error())
            else:
                # Message successfully received
                print(f"Received message: key={msg.key().decode('utf-8') if msg.key() else None}, "
                      f"value={msg.value().decode('utf-8')}")

    except KeyboardInterrupt:
        print("\nShutting down consumer.")

    finally:
        consumer.close()

if __name__ == "__main__":
    main()
