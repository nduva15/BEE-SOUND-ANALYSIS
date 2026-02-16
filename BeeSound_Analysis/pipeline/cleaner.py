"""
BEESOUND ANALYSIS - Noise Reduction Module
Removes environmental noise (wind, traffic, rain) from recordings.
"""

import numpy as np
import librosa
from scipy import signal

class AudioCleaner:
    """
    The Translator: Cleans raw field recordings before AI analysis.
    Based on the Bioacoustics package methodology.
    """
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        
    def remove_noise(self, audio, noise_profile=None):
        """
        Apply spectral subtraction to remove background noise.
        
        Args:
            audio: Input audio signal (numpy array)
            noise_profile: Optional noise sample for reference
            
        Returns:
            Cleaned audio signal
        """
        # Compute STFT
        D = librosa.stft(audio)
        magnitude, phase = librosa.magphase(D)
        
        # Estimate noise floor (bottom 10% of energy)
        if noise_profile is None:
            noise_floor = np.percentile(magnitude, 10, axis=1, keepdims=True)
        else:
            noise_D = librosa.stft(noise_profile)
            noise_floor = np.abs(noise_D).mean(axis=1, keepdims=True)
        
        # Spectral subtraction
        clean_magnitude = np.maximum(magnitude - noise_floor, 0)
        
        # Reconstruct signal
        clean_D = clean_magnitude * phase
        clean_audio = librosa.istft(clean_D)
        
        return clean_audio
    
    def apply_bandpass_filter(self, audio, low_freq=100, high_freq=8000):
        """
        Apply bandpass filter to focus on bee frequency range.
        
        Bee sounds are typically between 100Hz - 8kHz.
        This removes ultra-low rumble and high-frequency hiss.
        
        Args:
            audio: Input audio signal
            low_freq: Lower cutoff frequency in Hz
            high_freq: Upper cutoff frequency in Hz
            
        Returns:
            Filtered audio signal
        """
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_audio = signal.filtfilt(b, a, audio)
        
        return filtered_audio
    
    def clean(self, audio, apply_bandpass=True, remove_background=True):
        """
        Full cleaning pipeline.
        
        Args:
            audio: Input audio signal
            apply_bandpass: Whether to apply frequency filtering
            remove_background: Whether to apply noise reduction
            
        Returns:
            Cleaned audio signal
        """
        cleaned = audio.copy()
        
        if apply_bandpass:
            cleaned = self.apply_bandpass_filter(cleaned)
        
        if remove_background:
            cleaned = self.remove_noise(cleaned)
        
        return cleaned

if __name__ == "__main__":
    print("🔬 Audio Cleaner Module")
    print("   - Bandpass filter: 100Hz - 8kHz (bee frequency range)")
    print("   - Spectral subtraction for noise removal")
