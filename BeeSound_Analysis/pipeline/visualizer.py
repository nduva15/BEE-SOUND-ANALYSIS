"""
BEESOUND ANALYSIS - Spectrogram Visualization Module
Generates "Sound Pictures" for human inspection and AI input.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

class SpectrogramVisualizer:
    """
    Turns sound into visual representations.
    Helps humans see the buzz pattern and provides input for CNN models.
    """
    
    def __init__(self, sample_rate=22050, n_mels=128, n_fft=2048, hop_length=512):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def generate_mel_spectrogram(self, audio):
        """
        Generate mel-scaled spectrogram.
        
        Args:
            audio: Input audio signal (numpy array)
            
        Returns:
            Mel spectrogram in dB scale
        """
        # Compute mel spectrogram
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Convert to dB scale
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        return S_dB
    
    def save_spectrogram(self, audio, output_path, title="Bee Sound Signature"):
        """
        Generate and save spectrogram image.
        
        Args:
            audio: Input audio signal
            output_path: Where to save the image
            title: Plot title
        """
        S_dB = self.generate_mel_spectrogram(audio)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            S_dB,
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
            hop_length=self.hop_length,
            cmap='viridis'
        )
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path
    
    def visualize_from_file(self, audio_path, output_path=None):
        """
        Load audio file and generate spectrogram.
        
        Args:
            audio_path: Path to audio file
            output_path: Where to save image (auto-generated if None)
            
        Returns:
            Path to saved spectrogram image
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"data/processed_spectrograms/{base_name}_spectrogram.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create spectrogram
        title = f"Bee Sound: {os.path.basename(audio_path)}"
        self.save_spectrogram(y, output_path, title)
        
        return output_path

if __name__ == "__main__":
    print("📸 Spectrogram Visualizer Module")
    print("   - Mel-scaled frequency representation")
    print("   - Optimized for bee acoustic signatures (100Hz - 8kHz)")
