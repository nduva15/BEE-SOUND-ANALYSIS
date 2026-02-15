import urllib.request
import os

# 1. Create a folder to keep things tidy
# Standardizing path to match project structure
target_dir = "data/raw_audio"
if not os.path.exists(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    print(f"📁 Created folder: '{target_dir}'")

# 2. These are direct links to open-source bee sounds (verified safe)
# Note: Using .ogg links but librosa handles these fine. 
# We'll save them as .ogg or .wav depending on content.
urls = {
    "healthy_hive.ogg": "https://upload.wikimedia.org/wikipedia/commons/8/87/Honey_bee_sound.ogg",
    "queen_piping.ogg": "https://upload.wikimedia.org/wikipedia/commons/6/69/Queen_bee_piping.ogg"
}

print("⬇️  Starting download... this might take a few seconds.")

# 3. Download each file
for filename, link in urls.items():
    dest_path = os.path.join(target_dir, filename)
    print(f"   Downloading {filename}...")
    try:
        # User-agent might be needed for some wikimedia links
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(link, dest_path)
        print(f"   ✅ Saved: {dest_path}")
    except Exception as e:
        print(f"   ❌ Error downloading {filename}: {e}")

print("\n🎉 Done! You now have real data to test with.")
