from vosk import Model, KaldiRecognizer
import sounddevice as sd
import json

# Path to your Vosk model
MODEL_PATH = "vosk_model/vosk-model-en-us-0.22"

def test_vosk():
    try:
        print("Loading Vosk model...")
        model = Model(MODEL_PATH)  # Load the Vosk model
        recognizer = KaldiRecognizer(model, 16000)  # Initialize the recognizer with 16 kHz sample rate

        # Open a sounddevice InputStream
        print("Initializing microphone...")
        with sd.InputStream(samplerate=16000, channels=1, dtype="int16") as stream:
            print("Speak into the microphone. Press Ctrl+C to stop.")
            while True:
                # Read 4000 samples (~0.25 seconds) from the stream
                data = stream.read(4000)[0]  # Extract the data part of the tuple
                data_bytes = data.tobytes()  # Convert NumPy array to bytes
                if recognizer.AcceptWaveform(data_bytes):
                    result = json.loads(recognizer.Result())
                    if result.get("text"):
                        print(f"Recognized text: {result['text']}")
                else:
                    partial_result = json.loads(recognizer.PartialResult())
                    if partial_result.get("partial"):
                        print(f"Partial result: {partial_result['partial']}")

    except KeyboardInterrupt:
        print("\nTest stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_vosk()
