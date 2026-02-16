# 🚀 Quick Start Guide

## Running the Analysis

From the BeeSound_Analysis directory, run:

```bash
# Make sure you're in the right directory
cd c:\Users\aggym\Desktop\BEE-SOUND-ANALYSIS\BeeSound_Analysis

# Run analysis on the sample audio
python tools/run_analysis.py --input data/raw_audio/recording.wav
```

## Alternative: Run from Parent Directory

If you're in the parent BEE-SOUND-ANALYSIS directory:

```bash
python BeeSound_Analysis/tools/run_analysis.py --input BeeSound_Analysis/data/raw_audio/recording.wav
```

## Installation Steps

1. **Navigate to the project:**
   ```bash
   cd c:\Users\aggym\Desktop\BEE-SOUND-ANALYSIS\BeeSound_Analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download sample data:**
   ```bash
   python tools/download_data.py
   ```

4. **Run analysis:**
   ```bash
   python tools/run_analysis.py --input data/raw_audio/recording.wav
   ```

## Troubleshooting

### "Module not found" errors
Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### "File not found" errors
Check you're in the correct directory:
```bash
# Should show: C:\Users\aggym\Desktop\BEE-SOUND-ANALYSIS\BeeSound_Analysis
pwd
```

### Test individual modules
```bash
python -m pipeline.segmenter
python -m models.species_id
```
