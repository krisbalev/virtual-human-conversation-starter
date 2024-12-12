import pygame
import pyttsx3
import threading
from confluent_kafka import Consumer, KafkaException
import json

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("Virtual Human")
clock = pygame.time.Clock()

# Load images
human_idle = pygame.image.load("human_idle.png")
human_speaking = pygame.image.load("human_speaking.png")

# Resize images for consistency
human_idle = pygame.transform.scale(human_idle, (400, 400))
human_speaking = pygame.transform.scale(human_speaking, (400, 400))

# Text-to-Speech setup
tts_engine = pyttsx3.init()
tts_lock = threading.Lock()  # Lock to manage access to TTS

def speak(text):
    """Speak the given text aloud with thread safety."""
    with tts_lock:
        tts_engine.say(text)
        tts_engine.runAndWait()

# Kafka setup
KAFKA_BROKER = 'localhost:9092'
KAFKA_TOPIC = 'start_conversation'

consumer_config = {
    'bootstrap.servers': KAFKA_BROKER,
    'group.id': 'virtual_human_group',
    'auto.offset.reset': 'latest'
}

# Kafka Consumer
try:
    consumer = Consumer(consumer_config)
    consumer.subscribe([KAFKA_TOPIC])
except KafkaException as e:
    print(f"Error initializing Kafka Consumer: {e}")
    consumer = None

# Variables for animation
is_speaking = False
response_text = ""
font = pygame.font.Font(None, 36)

def listen_to_kafka():
    """Listen for Kafka messages and trigger responses."""
    global is_speaking, response_text

    if not consumer:
        print("Kafka Consumer not initialized. Exiting Kafka listener.")
        return

    while True:
        try:
            msg = consumer.poll(1.0)  # Poll for messages
            if msg is None:
                continue
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue

            # Process the received message
            event_data = json.loads(msg.value().decode('utf-8'))
            print(f"Received from Kafka: {event_data}")

            # Handle the cues from the event data
            if event_data.get("start_conversation"):
                cue_type = event_data.get("cue_type", "none")
                if cue_type == "verbal":
                    response_text = ""
                elif cue_type == "nonverbal":
                    response_text = "Hey there, Kristiyan!"
                elif cue_type == "both":
                    response_text = "Hello! How are you doing today?"
                else:
                    response_text = ""
                # Trigger speaking animation
                threading.Thread(target=simulate_speaking, args=(response_text,)).start()
            else:
                response_text = "Goodbye for now!"

            print(f"Respoded with: {response_text}")
        except Exception as e:
            print(f"Error during Kafka polling: {e}")

def simulate_speaking(response):
    """Simulate speaking animation."""
    global is_speaking
    is_speaking = True
    speak(response)
    is_speaking = False

# Start Kafka listener in a separate thread
if consumer:
    kafka_thread = threading.Thread(target=listen_to_kafka, daemon=True)
    kafka_thread.start()

# Main loop
running = True
while running:
    screen.fill((255, 255, 255))  # White background

    # Display appropriate image
    if is_speaking:
        screen.blit(human_speaking, (250, 100))
    else:
        screen.blit(human_idle, (250, 100))

    # Display response text
    response_surface = font.render(response_text, True, (0, 0, 0))
    screen.blit(response_surface, (50, 500))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS

pygame.quit()

# Close the Kafka consumer if initialized
if consumer:
    consumer.close()
