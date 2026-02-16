"""
BEESOUND ANALYSIS - Kaggle Compute Runner
Loads large HDF5 datasets and runs analysis in-memory.
Designed for: NUHIVE, TBON, SBCM, BAD datasets.
"""

import pandas as pd
import numpy as np
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from run_analysis import analyze_audio
except ImportError:
    try:
        from tools.run_analysis import analyze_audio
    except ImportError:
        # Fallback for relative run
        sys.path.append(os.path.dirname(__file__))
        from run_analysis import analyze_audio

def run_on_hdf5(h5_path, limit=None):
    """
    Process HDF5 dataset row by row without crashing RAM.
    """
    if not os.path.exists(h5_path):
        print(f"❌ Error: HDF5 file not found: {h5_path}")
        return

    print(f"📂 Opening Dataset: {h5_path}")
    
    try:
        with pd.HDFStore(h5_path, mode='r') as store:
            keys = store.keys()
            print(f"🔑 Keys found: {keys}")
            
            if not keys:
                print("⚠️  No keys found in HDF5 file.")
                return
            
            # Auto-detect key (usually /bee_audio or similar)
            target_key = keys[0]
            print(f"🎯 Using table: {target_key}")
            
            # Use iterator to load chunks
            iterator = store.select(target_key, chunksize=1, iterator=True)
            
            count = 0
            for chunk in iterator:
                if limit and count >= limit:
                    break
                
                # Each chunk is a DataFrame with 1 row
                row = chunk.iloc[0]
                
                # Try to find audio column
                audio_data = None
                label = f"Sample_{count}"
                
                # Common column names in research datasets
                possible_cols = ['audio', 'samples', 'signal', 'wav']
                for col in possible_cols:
                    if col in row:
                        audio_data = row[col]
                        break
                
                # If no audio column found, maybe the row itself is the array? (unlikely for DF)
                if audio_data is None:
                    # Fallback: Assume first column matches
                    if len(row) > 0:
                        audio_data = row[0] 
                
                if isinstance(audio_data, (np.ndarray, list)):
                    # Ensure float format
                    audio_data = np.array(audio_data, dtype=np.float32)
                    
                    print(f"\n🎧 Processing Sample #{count}...")
                    analyze_audio(
                        audio_data=audio_data, 
                        sample_rate=22050,  # Assume standard SR or read from col
                        save_spectrogram=False, 
                        verbose=True
                    )
                    count += 1
                else:
                    print(f"⚠️  Skipping row {count}: No audio array found.")

    except Exception as e:
        print(f"❌ Error processing HDF5: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeeSound Kaggle Runner")
    parser.add_argument("--input", type=str, required=True, help="Path to .h5 file")
    parser.add_argument("--limit", type=int, default=5, help="Number of samples to process")
    
    args = parser.parse_args()
    
    run_on_hdf5(args.input, args.limit)
