from confluent_kafka import Consumer
import json
import requests
import asyncio
import websockets
import time

# Kafka configuration
KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'd-id-agent-group',
    'auto.offset.reset': 'earliest'
}

# Kafka topic
TOPIC_NAME = "start_conversation"

# D-ID API configuration
D_ID_API_URL = "https://api.d-id.com/talks"
D_ID_API_KEY = "bG9sLm1hc3RlcjI5MDhAZ21haWwuY29t:mbmyIyZRrBxyzDFaMFQtZ"  # Replace with your actual API key

# WebSocket server URL
WEBSOCKET_SERVER_URL = "ws://localhost:8000"


async def send_to_websocket(message):
    """
    Sends a message to the WebSocket server.
    """
    try:
        async with websockets.connect(WEBSOCKET_SERVER_URL) as websocket:
            await websocket.send(json.dumps(message))
            print(f"Sent to WebSocket: {message}")
    except Exception as e:
        print(f"Error sending to WebSocket: {e}")


def send_to_did_api(cues):
    """
    Sends cues to the D-ID API to create a conversation task.
    """
    payload = {
        "script": {
            "type": "text",
            "input": f"Start a conversation based on these nonverbal cues: {', '.join(cues)}."
        },
        "config": {
            "stitch": True,
            "driver_url": "https://d-id-demo.s3.us-west-2.amazonaws.com/sample_avatars/ron.png"
        }
    }

    headers = {
        "Authorization": f"Basic {D_ID_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(D_ID_API_URL, json=payload, headers=headers)
        if response.status_code == 201:
            result = response.json()
            task_id = result['id']
            print(f"D-ID API Task Created: {task_id}")
            return task_id
        else:
            print(f"Error from D-ID API: {response.status_code}, {response.text}")
    except Exception as e:
        print(f"Exception when calling D-ID API: {e}")
    return None


def poll_did_task_status(task_id):
    """
    Polls the D-ID API for task completion and returns the result URL when ready.
    """
    status_url = f"{D_ID_API_URL}/{task_id}"

    headers = {
        "Authorization": f"Basic {D_ID_API_KEY}",
        "Content-Type": "application/json"
    }

    print(f"Polling D-ID API for task status: {task_id}")

    while True:
        try:
            response = requests.get(status_url, headers=headers)
            if response.status_code == 200:
                result = response.json()
                print(f"Task Status: {result['status']}")

                if result['status'] == "done":
                    print("Task completed successfully!")
                    return result.get('result_url', None)
                elif result['status'] == "error":
                    print(f"Task failed: {result.get('message', 'No error message provided')}")
                    return None
            else:
                print(f"Error polling task status: {response.status_code}, {response.text}")

        except Exception as e:
            print(f"Exception while polling task status: {e}")
            return None

        time.sleep(2)  # Poll every 2 seconds


def process_message(data):
    """
    Processes the Kafka message and triggers the D-ID API workflow.
    """
    cues = data.get("cues", [])
    print(f"Processing cues: {cues}")

    # Step 1: Send cues to D-ID API and get task ID
    task_id = send_to_did_api(cues)
    if not task_id:
        print("Failed to create D-ID task.")
        return

    # Step 2: Poll for task completion
    result_url = poll_did_task_status(task_id)
    if result_url:
        print(f"D-ID Result URL: {result_url}")

        # Step 3: Send the result URL to the WebSocket frontend
        asyncio.run(send_to_websocket({"type": "result", "url": result_url}))
    else:
        print("No result URL obtained from D-ID API.")


def consume_messages():
    """
    Consumes messages from the Kafka topic and processes them.
    """
    consumer = Consumer(KAFKA_CONFIG)
    consumer.subscribe([TOPIC_NAME])

    print("Kafka consumer is listening to the topic...")

    try:
        while True:
            msg = consumer.poll(1.0)  # Poll for a message
            if msg is None:
                continue  # No message received
            if msg.error():
                print(f"Kafka consumer error: {msg.error()}")
                continue

            # Decode the Kafka message
            message = json.loads(msg.value().decode('utf-8'))
            print(f"Received Kafka message: {message}")

            # Process the message
            if message.get("start_conversation"):
                process_message(message)

    except KeyboardInterrupt:
        print("\nKafka consumer interrupted by user.")
    finally:
        # Ensure the consumer is properly closed
        consumer.close()
        print("Kafka consumer shut down.")


if __name__ == "__main__":
    consume_messages()
