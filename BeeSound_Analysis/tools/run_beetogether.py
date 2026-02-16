"""
BEESOUND ANALYSIS - Master BeeTogether Integrator
Processes the 'Big 4' (NUHIVE, TBON, SBCM, BAD) safely on Kaggle.
"""

import pandas as pd
import numpy as np
import os
import sys

# 1. Setup paths to find run_analysis
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_root)

try:
    from tools.run_analysis import analyze_audio
except ImportError:
    from run_analysis import analyze_audio

def process_combined_datasets(base_path="/kaggle/input/beetogether/", limit_per_file=5):
    """
    Implements your concatenation logic but processes row-by-row
    to prevent Kaggle RAM crashes.
    """
    files = ["NUHIVE.h5", "TBON.h5", "SBCM.h5", "BAD.h5"]
    
    print("="*70)
    print("🐝 BEETOGETHER MASTER INTEGRATOR")
    print("="*70)

    for file_name in files:
        full_path = os.path.join(base_path, file_name)
        
        if not os.path.exists(full_path):
            print(f"❌ Missing: {file_name} at {full_path}")
            continue
            
        print(f"\n📂 Processing Dataset: {file_name}")
        
        try:
            # We use an iterator to avoid loading all 28GB at once
            with pd.HDFStore(full_path, mode='r') as store:
                if '/bee_audio' not in store.keys():
                    print(f"   ⚠️ Key '/bee_audio' not found. Available: {store.keys()}")
                    continue
                
                # Fetch a chunk
                df_chunk = store.select('bee_audio', start=0, stop=limit_per_file)
                
                for i, row in df_chunk.iterrows():
                    # Identify the audio column (usually 'audio' or 'samples')
                    audio_samples = None
                    for col in ['audio', 'samples', 'signal', 'wav']:
                        if col in row:
                            audio_samples = row[col]
                            break
                    
                    if audio_samples is not None:
                        print(f"   🎧 Analyzing {file_name} Sample #{i}...")
                        analyze_audio(
                            audio_data=np.array(audio_samples, dtype=np.float32),
                            sample_rate=22050,
                            save_spectrogram=False,
                            verbose=True
                        )
                    else:
                        print(f"   ⚠️  Row {i} has no recognizable audio column.")
                        
        except Exception as e:
            print(f"   ❌ Error processing {file_name}: {e}")

if __name__ == "__main__":
    process_combined_datasets()
