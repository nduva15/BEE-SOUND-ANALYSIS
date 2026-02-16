"""
BEESOUND ANALYSIS - Audio Segmentation Module
Slices raw audio into 2-second windows for AI processing.
"""

import numpy as np
import librosa
import os

class AudioSegmenter:
    """
    The Science: AI models were trained on 2-second clips.
    This class ensures all audio is properly windowed.
    """
    
    def __init__(self, window_size=2.0, overlap=0.5, sample_rate=22050):
        """
        Args:
            window_size: Duration of each segment in seconds (default: 2.0)
            overlap: Overlap between segments in seconds (default: 0.5)
            sample_rate: Target sample rate in Hz (default: 22050)
        """
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        
    def segment_audio(self, file_path):
        """
        Slice long audio into 2-second overlapping windows.
        
        Logic:
        - 1 minute of audio → ~120 analyzable samples
        - Overlap ensures we don't miss events at boundaries
        
        Returns:
            List of audio segments (numpy arrays)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load audio
        y, sr = librosa.load(file_path, sr=self.sample_rate)
        
        # Calculate window parameters
        window_samples = int(self.window_size * self.sample_rate)
        hop_samples = int((self.window_size - self.overlap) * self.sample_rate)
        
        segments = []
        timestamps = []
        
        # Create overlapping segments
        for start in range(0, len(y) - window_samples + 1, hop_samples):
            segment = y[start : start + window_samples]
            segments.append(segment)
            timestamps.append(start / self.sample_rate)
        
        return segments, timestamps
    
    def get_segment_count(self, duration_seconds):
        """
        Calculate how many segments will be generated from a given duration.
        
        Args:
            duration_seconds: Total audio duration in seconds
            
        Returns:
            Number of segments
        """
        hop_duration = self.window_size - self.overlap
        return int((duration_seconds - self.window_size) / hop_duration) + 1

if __name__ == "__main__":
    # Test the segmenter
    segmenter = AudioSegmenter()
    
    # Example: 60 seconds of audio
    expected_segments = segmenter.get_segment_count(60)
    print(f"📊 60 seconds of audio → {expected_segments} segments")
    print(f"   Window: {segmenter.window_size}s, Overlap: {segmenter.overlap}s")
