"""
BEESOUND ANALYSIS - Data Acquisition Module
Downloads the "Golden Standard" audio samples for testing and validation.
"""

import urllib.request
import os
import sys

def download_file(url, destination):
    """Download a file with progress indication."""
    try:
        print(f"   📥 Downloading from: {url}")
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, destination)
        file_size = os.path.getsize(destination) / 1024  # KB
        print(f"   ✅ Saved: {destination} ({file_size:.1f} KB)")
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("="*60)
    print("🐝 BEESOUND ANALYSIS - Data Acquisition")
    print("="*60)
    
    # Create directory structure
    target_dir = "data/raw_audio"
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 Target directory: {target_dir}\n")
    
    # Dataset URLs - Open-source bee audio samples
    datasets = {
        "healthy_hive.wav": "https://freesound.org/data/previews/411/411749_7517113-lq.mp3",
        "bee_buzzing.wav": "https://freesound.org/data/previews/265/265828_4486188-lq.mp3",
        "hive_activity.wav": "https://freesound.org/data/previews/398/398513_7517113-lq.mp3"
    }
    
    print("⬇️  Starting download sequence...\n")
    
    success_count = 0
    for filename, url in datasets.items():
        dest_path = os.path.join(target_dir, filename)
        print(f"[{success_count + 1}/{len(datasets)}] {filename}")
        if download_file(url, dest_path):
            success_count += 1
        print()
    
    print("="*60)
    if success_count == len(datasets):
        print("🎉 SUCCESS! All datasets downloaded.")
        print(f"📂 Location: {os.path.abspath(target_dir)}")
    else:
        print(f"⚠️  Partial success: {success_count}/{len(datasets)} files downloaded.")
    print("="*60)
    
    return success_count == len(datasets)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
