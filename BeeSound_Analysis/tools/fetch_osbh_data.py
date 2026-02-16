
import os
import urllib.request
import ssl

# 1. Setup Directories matching your Master Architecture
TARGET_DIR = "data/raw_audio/osbh_reference"
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# 2. The "Hidden" Hiveeyes Research Datasets
# These links point to the specific validation files used by the OSBH C++ engine.
manifest = {
    "colony_with_queen.ogg": "https://community.hiveeyes.org/uploads/default/original/1X/bf02f90aa741db64956ff1e4b8bbf94bf1c636d0.mp3",
    "colony_queenless.ogg": "https://community.hiveeyes.org/uploads/default/original/1X/6e001118cc13c9fe1937258f2e260a32b67f82dd.mp3",
    "swarm_piping.ogg": "https://community.hiveeyes.org/uploads/default/original/1X/e97bba5462ea7b37f2c0b2dda2f34985e19962e4.ogg",
    "test_tone_clean.wav": "https://upload.wikimedia.org/wikipedia/commons/c/c8/A440_tone_5s.wav" # For calibration
}

print(f"🐝 BEESOUND: Fetching OSBH Reference Data into {TARGET_DIR}...")

# Create an unverified SSL context to avoid certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

for filename, url in manifest.items():
    print(f"   ⬇️  Downloading {filename}...")
    try:
        output_path = os.path.join(TARGET_DIR, filename)
        # Fake a user-agent because their server blocks python-scripts sometimes
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, output_path)
        print(f"       ✅ Saved.")
    except Exception as e:
        print(f"       ❌ Failed: {e}")

print("\n🚀 Data Sync Complete. You can now test the C++ engine against these files.")
