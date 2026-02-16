"""
BEESOUND ANALYSIS - Kaggle Data Loader
Fetches large datasets (28GB+) on demand without crashing local disk.
Requires: kaggle.json in ~/.kaggle/
"""

import os
import argparse

def fetch_dataset(dataset_slug, file_name=None, target_dir="data/raw_audio/"):
    """
    Download specific files or entire datasets from Kaggle.
    
    Args:
        dataset_slug (str): Kaggle dataset identifier (e.g. 'nduva15/beetogether-audio')
        file_name (str, optional): Specific file to download (e.g. 'queen_piping.h5')
        target_dir (str): Local destination folder
    """
    try:
        import kaggle
    except ImportError:
        print("❌ Error: 'kaggle' library not installed.")
        print("   Run: pip install kaggle")
        return

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        print(f"📂 Created directory: {target_dir}")

    print(f"⬇️  Connecting to Kaggle API...")
    
    try:
        # Check authentication first
        kaggle.api.authenticate()
        print("✅ Authenticated successfully.")
        
        if file_name:
            print(f"⬇️  Downloading specific file: {file_name}...")
            kaggle.api.dataset_download_file(
                dataset_slug,
                encrypted_file_name=file_name,
                path=target_dir,
                force=True,
                quiet=False
            )
            print(f"✅ Downloaded: {os.path.join(target_dir, file_name)}")
        else:
            print(f"⚠️  WARNING: Downloading FULL dataset ({dataset_slug})...")
            print("   This might take a while for large files (28GB+).")
            kaggle.api.dataset_download_files(
                dataset_slug,
                path=target_dir,
                unzip=True,
                quiet=False
            )
            print(f"✅ Full dataset downloaded to: {target_dir}")
            
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        print("   Common fixes:")
        print("   1. Check if ~/.kaggle/kaggle.json exists")
        print("   2. Verify dataset slug is correct")
        print("   3. Check internet connection")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeeSound Kaggle Loader")
    parser.add_argument("--dataset", type=str, required=True, help="Kaggle Dataset Slug (user/dataset)")
    parser.add_argument("--file", type=str, help="Specific file to download (optional)")
    parser.add_argument("--output", type=str, default="data/raw_audio/", help="Output directory")
    
    args = parser.parse_args()
    
    fetch_dataset(args.dataset, args.file, args.output)
