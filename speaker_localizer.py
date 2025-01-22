import numpy as np
import pyroomacoustics as pra

class SpeakerLocalizer:
    def __init__(self, mic_positions, sample_rate=16000, nfft=256):
        """
        Initializes the microphone array for DoA estimation.
        :param mic_positions: 3D positions of microphones as a NumPy array of shape (3, num_mics).
        :param sample_rate: Audio sampling rate in Hz.
        :param nfft: Number of FFT points for MUSIC algorithm.
        """
        # Ensure mic_positions is a 2D array of shape (3, num_mics)
        if mic_positions.shape[0] != 3:
            raise ValueError("mic_positions must have shape (3, num_mics)")

        self.mic_positions = mic_positions  # Save mic positions directly
        self.sample_rate = sample_rate
        self.nfft = nfft

        # Number of microphones
        self.num_mics = self.mic_positions.shape[1]

        # Initialize MUSIC DoA estimation
        self.doa = pra.doa.MUSIC(
            R=self.mic_positions,  # Use mic positions directly
            L=self.num_mics,       # Number of microphones
            fs=self.sample_rate,
            nfft=self.nfft,
            c=pra.constants.get('c')  # Speed of sound
        )
    
    def get_direction(self, signal):
        """
        Estimate the direction of the sound source.
        :param signal: Multi-channel audio signal (NumPy array) of shape (num_channels, num_samples).
        :return: Direction of arrival (azimuth angle in degrees).
        """
        # Ensure the signal has the correct dimensions
        if signal.shape[0] != self.num_mics:
            raise ValueError(f"Signal must have {self.num_mics} channels, "
                             f"but got {signal.shape[0]} channels.")

        # Pass the signal to the DoA estimator
        self.doa.locate_sources(signal)

        # Return the first estimated direction (in degrees)
        if len(self.doa.azimuth_recon) > 0:
            return self.doa.azimuth_recon[0] * 180 / np.pi  # Convert radians to degrees
        else:
            raise ValueError("No direction of arrival found. Check your signal and setup.")
