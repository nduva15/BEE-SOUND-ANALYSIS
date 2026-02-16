"""
BEESOUND ANALYSIS - Event Detector
Stage 3: Is there an Emergency?
Model: Frequency-domain pattern matching for queen piping
Target Recall: 98.1%
"""

import numpy as np
import librosa
from scipy import signal

class EventDetector:
    """
    Detector: Queen Piping & Swarming Signals
    Based on beepiping repository (Fourer & Orlorwska, DCASE 2022).
    """
    
    def __init__(self):
        # Queen piping characteristics
        self.piping_freq_range = (300, 500)  # Hz
        self.piping_duration_min = 0.1       # seconds
        self.piping_threshold = 0.7          # confidence threshold
        
    def detect_piping(self, audio, sample_rate=22050):
        """
        Detect queen piping signals.
        
        Queen piping is a high-pitched "toot" sound (300-500 Hz)
        that indicates:
        - Multiple queens in hive (pre-swarm)
        - Queen emergence imminent
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: {'detected': bool, 'confidence': float, 'timestamps': list}
        """
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024)
        
        # Find frequency bins corresponding to piping range
        freq_mask = (f >= self.piping_freq_range[0]) & (f <= self.piping_freq_range[1])
        piping_band = Sxx[freq_mask, :]
        
        # Compute energy in piping frequency band
        piping_energy = np.sum(piping_band, axis=0)
        
        # Normalize
        if np.max(piping_energy) > 0:
            piping_energy = piping_energy / np.max(piping_energy)
        
        # Detect peaks (potential piping events)
        threshold = self.piping_threshold
        peaks, properties = signal.find_peaks(
            piping_energy,
            height=threshold,
            distance=int(self.piping_duration_min * len(t) / t[-1])
        )
        
        # Convert peak indices to timestamps
        timestamps = [t[peak] for peak in peaks]
        
        # Calculate confidence based on peak strength
        if len(peaks) > 0:
            confidence = float(np.mean(properties['peak_heights']))
            detected = True
        else:
            confidence = 0.0
            detected = False
        
        return {
            'detected': detected,
            'confidence': confidence,
            'event_count': len(peaks),
            'timestamps': timestamps,
            'frequency_range': self.piping_freq_range
        }
    
    def detect_hissing(self, audio, sample_rate=22050):
        """
        Detect defensive hissing (high-frequency broadband noise).
        
        Indicates:
        - Hive disturbance
        - Defensive behavior
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: {'detected': bool, 'confidence': float}
        """
        # Compute spectral flatness (measure of noise-like quality)
        flatness = librosa.feature.spectral_flatness(y=audio)
        
        # High flatness = noise-like = potential hissing
        mean_flatness = np.mean(flatness)
        
        # Hissing is typically > 0.5 flatness
        detected = mean_flatness > 0.5
        confidence = float(min(mean_flatness / 0.5, 1.0))
        
        return {
            'detected': detected,
            'confidence': confidence,
            'flatness': float(mean_flatness)
        }
    
    def analyze(self, audio, sample_rate=22050):
        """
        Run full event detection pipeline.
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: Combined results from all detectors
        """
        piping = self.detect_piping(audio, sample_rate)
        hissing = self.detect_hissing(audio, sample_rate)
        
        # Determine overall alert level
        if piping['detected']:
            alert_level = 'CRITICAL'
            alert_message = 'Queen piping detected - Swarm imminent!'
        elif hissing['detected']:
            alert_level = 'WARNING'
            alert_message = 'Defensive behavior detected'
        else:
            alert_level = 'NORMAL'
            alert_message = 'No emergency signals detected'
        
        return {
            'alert_level': alert_level,
            'alert_message': alert_message,
            'piping': piping,
            'hissing': hissing
        }

if __name__ == "__main__":
    print("🚨 Event Detector")
    print("   Target: 98.1% recall for piping detection")
    print("   Signals: Queen piping (300-500Hz), Defensive hissing")
