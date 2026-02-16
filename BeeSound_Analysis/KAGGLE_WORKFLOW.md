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

---

**Happy Cloud Computing!** 🐝
