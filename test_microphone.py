import sounddevice as sd
import queue
import json
import time
from vosk import Model, KaldiRecognizer

class VoiceDetector:
    def __init__(self, model_path="vosk_model/vosk-model-en-us-0.22"):
        """
        Initialize the Vosk model and audio stream for speech recognition.

        :param model_path: Path to the pre-trained Vosk model directory.
        """
        try:
            self.model = Model(model_path)
            print(f"Loaded Vosk model from '{model_path}'.")
        except Exception as e:
            print(f"Failed to load model from '{model_path}': {e}")
            raise e

        try:
            self.recognizer = KaldiRecognizer(self.model, 16000)  # Initialize recognizer with sample rate
            self.recognizer.SetWords(True)  # Optional: Enables word-level timestamps
            print("KaldiRecognizer initialized.")
        except Exception as e:
            print(f"Failed to initialize KaldiRecognizer: {e}")
            raise e

        self.audio_queue = queue.Queue()

        try:
            self.stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="int16",
                callback=self.audio_callback
            )
            self.stream.start()
            print("Audio stream started.")
        except Exception as e:
            print(f"Failed to start audio stream: {e}")
            raise e

    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback to capture audio from the microphone and add it to the queue.

        :param indata: Captured audio data.
        :param frames: Number of audio frames.
        :param time_info: Time information.
        :param status: Stream status.
        """
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())  # Copy the data to avoid conflicts

    def detect_speech(self):
        """
        Process audio from the queue and detect spoken text.

        :return: Transcribed text or None if no speech is detected or if the text is "the".
        """
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            data_bytes = data.tobytes()  # Convert NumPy array to bytes for Vosk

            if self.recognizer.AcceptWaveform(data_bytes):
                # Final result available
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()
                print(f"Vosk Final Result: '{result.get('text', '').strip()}'")
                
                if text and text != "the":  # Only return if text is non-empty and not "the"
                    original_text = result.get("text", "").strip()
                    return original_text
                else:
                    print("Detected speech is 'the'; ignoring.")
            else:
                # Partial result available
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get("partial", "")
                if partial_text:
                    print(f"Vosk Partial Result: '{partial_text}'")
                # You can choose to handle partial results as needed

        return None  # No complete speech detected in the current buffer

    def stop(self):
        """
        Stop and close the audio stream safely.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()
            print("Audio stream stopped and closed.")

def main():
    # Path to your Vosk model directory
    MODEL_PATH = "vosk_model/vosk-model-en-us-0.22"

    # Initialize the VoiceDetector
    try:
        detector = VoiceDetector(model_path=MODEL_PATH)
    except Exception as e:
        print("Exiting due to initialization failure.")
        exit(1)

    print("VoiceDetector is running. Press Ctrl+C to stop.")

    try:
        while True:
            # Detect speech
            speech = detector.detect_speech()
            if speech:
                print(f"Detected Speech: '{speech}'")
            # Sleep briefly to prevent a tight loop and reduce CPU usage
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Stopping VoiceDetector.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        detector.stop()
        print("VoiceDetector has been stopped.")

if __name__ == "__main__":
    main()
