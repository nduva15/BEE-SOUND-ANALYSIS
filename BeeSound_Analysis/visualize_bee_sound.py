import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

def create_spectrogram(audio_path, output_image="hive_view.png"):
    """
    Turns sound into a 'Sound Picture' (Spectrogram).
    Helps humans see the buzz pattern!
    """
    if not os.path.exists(audio_path):
        print(f"❌ Error: File not found at {audio_path}")
        return

    print(f"📸 Generating picture for: {audio_path}")
    
    # Load the audio
    y, sr = librosa.load(audio_path)
    
    # Generate the Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Create the plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Bee Sound Signature: {os.path.basename(audio_path)}')
    plt.tight_layout()
    
    # Save it
    plt.savefig(output_image)
    print(f"✅ Success! Spectrogram saved as: {output_image}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Bee Sounds")
    parser.add_argument("--input", type=str, help="Path to audio file", default="data/raw_audio/healthy_hive.ogg")
    args = parser.parse_args()

    # Check if we should download sounds first
    if not os.path.exists(args.input):
        print(f"⚠️  Input file '{args.input}' not found.")
        print("💡 Pro-tip: Run 'python get_bee_sounds.py' first!")
    else:
        create_spectrogram(args.input)
