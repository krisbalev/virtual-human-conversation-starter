import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer

class VoiceDetector:
    def __init__(self, model_path="vosk_model/vosk-model-en-us-0.22"):
        """
        Initialize the Vosk model and audio stream for speech recognition.

        :param model_path: Path to the pre-trained Vosk model directory.
        """
        # Load the Vosk model
        self.model = Model(model_path)
        
        # Initialize the recognizer with a 16kHz sample rate
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        # Optional: Enable word-level timestamps for more detailed analysis
        self.recognizer.SetWords(True)
        
        # Create a queue for audio data
        self.audio_queue = queue.Queue()
        
        # Configure and start audio input stream
        self.stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype="int16",
            callback=self.audio_callback
        )
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        """
        Callback to capture audio from the microphone and add it to the queue.

        :param indata: Captured audio data (NumPy array).
        :param frames: Number of audio frames.
        :param time:   Current time information (ignored here).
        :param status: Stream status (e.g., errors or warnings).
        """
        if status:
            # Handle status as needed, e.g., logging warnings/errors
            pass
        
        # Copy the data to avoid concurrency issues
        self.audio_queue.put(indata.copy())

    def detect_speech(self):
        """
        Process audio from the queue and detect spoken text.

        :return: Transcribed text (string) or None if no valid speech is detected.
        """
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            data_bytes = data.tobytes()  # Convert from NumPy array to bytes for Vosk
            
            if self.recognizer.AcceptWaveform(data_bytes):
                # On final result
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip().lower()

                # --- Post-processing for filler words or unwanted prefixes ---
                # Example: remove leading "the " if it appears
                if text.startswith("the "):
                    text = text[4:].strip()  # remove the "the " prefix

                # Filter out known filler words like "the", "um", "uh"
                if text and text not in ["the", "um", "uh"]:
                    return text
            else:
                # On partial result (optional)
                partial_result = json.loads(self.recognizer.PartialResult())
                partial_text = partial_result.get("partial", "")
                pass

        return None

    def stop(self):
        """
        Stop and close the audio stream safely.
        """
        if self.stream:
            self.stream.stop()
            self.stream.close()
