"""
BEESOUND ANALYSIS - Simple Test Script (No Dependencies)
Test the project structure without heavy libraries
"""

import sys
import os

print("=" * 70)
print("🐝 BEESOUND ANALYSIS - Simple Test")
print("=" * 70)
print()

# Check directory structure
print("📂 Checking project structure...")
print()

required_dirs = [
    "data/raw_audio",
    "data/processed_spectrograms", 
    "data/datasets_metadata",
    "pipeline",
    "models",
    "tools",
    "firmware"
]

all_ok = True
for dir_path in required_dirs:
    exists = os.path.exists(dir_path)
    status = "✅" if exists else "❌"
    print(f"   {status} {dir_path}")
    if not exists:
        all_ok = False

print()
print("=" * 70)

if all_ok:
    print("✅ SUCCESS! Project structure is correctly set up.")
    print()
    print("📋 Next Steps:")
    print("   1. Install dependencies: pip install numpy pandas scipy matplotlib librosa soundfile scikit-learn")
    print("   2. Run: python tools/run_analysis.py --input data/raw_audio/recording.wav")
else:
    print("⚠️  Some directories are missing. Project may need setup.")

print("=" * 70)
