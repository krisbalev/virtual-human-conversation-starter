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
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, 16000)  # Initialize recognizer with sample rate
        self.audio_queue = queue.Queue()
        self.stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16", callback=self.audio_callback)
        self.stream.start()

    def audio_callback(self, indata, frames, time, status):
        """
        Callback to capture audio from the microphone and add it to the queue.
        :param indata: Captured audio data.
        :param frames: Number of audio frames.
        :param time: Current time information.
        :param status: Stream status.
        """
        if status:
            print(f"Audio status: {status}")
        self.audio_queue.put(indata.copy())  # Copy the data to avoid conflicts

    def detect_speech(self):
        """
        Process audio from the queue and detect spoken text.
        :return: Transcribed text or None if no speech is detected.
        """
        while not self.audio_queue.empty():
            data = self.audio_queue.get()
            data_bytes = data.tobytes()  # Convert NumPy array to bytes for Vosk
            if self.recognizer.AcceptWaveform(data_bytes):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "")
                if text:  # Only return if text is non-empty
                    return text
        return None

    def stop(self):
        """
        Stop and close the audio stream safely.
        """
        self.stream.stop()
        self.stream.close()
