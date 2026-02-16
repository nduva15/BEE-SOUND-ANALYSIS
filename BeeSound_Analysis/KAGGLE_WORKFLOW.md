# ☁️ Beekeeping in the Cloud: Computing on Kaggle

This guide explains how to run the full BeeSound Analysis pipeline on **Kaggle Kernels** or similar cloud environments without downloading the massive 28GB dataset to your local machine.

## 1. Why Compute on Kaggle?
-   **Free GPU**: Accelerates AI inference.
-   **Zero Downloads**: The data is already there (`/kaggle/input/`).
-   **No Disk Crash**: Your laptop stays safe.

## 2. Setup (Inside a Kaggle Notebook)

### Step 1: Clone the Repository
At the top of your Kaggle Notebook, run this cell to get the latest code:

```python
!git clone https://github.com/nduva15/BEE-SOUND-ANALYSIS.git
%cd BEE-SOUND-ANALYSIS/BeeSound_Analysis
!pip install -r requirements.txt
```

### Step 2: Locate Your Data
Kaggle mounts datasets at `/kaggle/input/`. Verify the path:

```python
import os
print(os.listdir("/kaggle/input/"))
# Example Output: ['beetogether-audio', 'nuhive-sample']
```

## 3. Run Analysis (Lazy Loading)

We have created a special tool `tools/run_kaggle.py` that reads the huge HDF5 files **row-by-row** without loading the whole file into RAM.

```python
# Run on the first 10 samples of the NUHIVE dataset
!python tools/run_kaggle.py --input /kaggle/input/beetogether-audio/NUHIVE.h5 --limit 10
```

### Expected Output
```text
📂 Opening Dataset: /kaggle/input/beetogether-audio/NUHIVE.h5
🔑 Keys found: ['/bee_audio']
🎯 Using table: /bee_audio

🎧 Processing Sample #0...
======================================================================
🐝 BEESOUND ANALYSIS PIPELINE
======================================================================

📊 Analyzing Raw Data: 2.0s @ 22050Hz

🔪 Segmenting audio...
✅ Generated 1 segments

🧹 Cleaning audio...
✅ Cleaned 1 segments

STAGE 1: Species Identification
   Species: Apis Mellifera (98.2%)

STAGE 2: Health State
   Colony State: Healthy

STAGE 3: Emergency Signals
   ✅ No emergency signals detected
```

## 4. Advanced Usage

### Analyze Multiple Files
You can write a simple loop in Python:

```python
import glob

# Find all HDF5 files
files = glob.glob("/kaggle/input/**/*.h5", recursive=True)

for f in files:
    print(f"🚀 Processing {f}...")
    !python tools/run_kaggle.py --input "{f}" --limit 5
```

## 4. Dataset Locations (The "Big 4")
The 28GB dataset is located at these exact paths on Kaggle:
- **NUHIVE**: `/kaggle/input/datasets/augustin23/beetogether/NUHIVE/NUHIVE`
- **BAD**: `/kaggle/input/datasets/augustin23/beetogether/BAD/BAD`
- **SBCM**: `/kaggle/input/datasets/augustin23/beetogether/SBCM/SBCM`
- **TBON**: `/kaggle/input/datasets/augustin23/beetogether/TBON/TBON`

## 5. Master Indexer (Before Training)
Run this in Kaggle to create a manifest of all 28GB of audio:

```python
import os
import pandas as pd
data_dirs = ["/kaggle/input/datasets/augustin23/beetogether/NUHIVE/NUHIVE", "/kaggle/input/datasets/augustin23/beetogether/BAD/BAD", "/kaggle/input/datasets/augustin23/beetogether/SBCM/SBCM", "/kaggle/input/datasets/augustin23/beetogether/TBON/TBON"]
manifest = []
for d in data_dirs:
    for root, _, files in os.walk(d):
        for f in files:
            if f.endswith(('.wav', '.ogg', '.mp3')):
                manifest.append({'path': os.path.join(root, f)})
pd.DataFrame(manifest).to_csv('bee_training_master.csv', index=False)
```

---

**Happy Cloud Training!** 🐝
