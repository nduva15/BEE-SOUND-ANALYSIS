import os
import pandas as pd
import sys

def fast_index():
    print("🚀 NITRO INDEXER v2.0 (Self-Healing)")
    print("======================================================================")
    
    # Check if we are in the right directory
    if not os.path.exists('tools/train_architecture.py'):
        print("❌ Error: Not in the project root folder.")
        print("   Run: %cd /kaggle/working/BEE-SOUND-ANALYSIS/BeeSound_Analysis")
        return

    # 1. Faster Search
    print("📂 Checking mounted datasets in /kaggle/input...")
    os.system('ls -d /kaggle/input/*/')
    
    print("\n🔍 Probing Kaggle Inputs (Exhaustive Search)...")
    # Use -iname for case-insensitivity and search for both wav and WAV
    os.system('find /kaggle/input -iname "*.wav" > raw_files.txt')
    
    if not os.path.exists('raw_files.txt') or os.path.getsize('raw_files.txt') == 0:
        print("❌ CRITICAL: No .wav files found in /kaggle/input.")
        return

    # 2. Reading with low-level file handler for better speed
    all_files = []
    with open('raw_files.txt', 'r') as f:
        for line in f:
            all_files.append(line.strip())
    
    total = len(all_files)
    print(f"✅ Found {total} recordings.")

    # 3. Labeling Logic
    print("🧠 Mapping Bioacoustic Labels...")
    keywords = ['queenless', 'alert', 'piping', 'nq', 'ql', 'no_queen']
    
    data = []
    for i, path in enumerate(all_files):
        label = 1 if any(k in path.lower() for k in keywords) else 0
        data.append({'file_path': path, 'label': label})
        
        if i % 50000 == 0 and i > 0:
            print(f"   Progress: {i}/{total} ({ (i/total)*100:.1f}%)")

    # 4. Memory-Efficient Save
    print("💾 Finalizing Manifest: train_manifest_labeled.csv")
    df = pd.DataFrame(data)
    df.to_csv('train_manifest_labeled.csv', index=False)
    
    # 5. Verification
    if os.path.exists('train_manifest_labeled.csv'):
        print(f"✨ SUCCESS: Manifest restored with {len(df)} samples.")
        print("🚀 You can now run: !python tools/train_architecture.py --epochs 1")
    else:
        print("❌ Error: Failed to save manifest.")

if __name__ == "__main__":
    fast_index()
