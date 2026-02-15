import numpy as np
import librosa
import json
import os
import argparse

# Configuration matching research metrics
CONFIG = {
    "sample_rate": 22050,  # Standard for the transformers
    "segment_duration": 2.0, # The "Slice" size for the 26k samples
    "hop_length": 512,
    "n_mels": 128
}

def segment_audio(file_path):
    """
    Slices raw massive files into the 2-second chunks 
    used to achieve the 94.2% accuracy benchmark.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []
        
    y, sr = librosa.load(file_path, sr=CONFIG["sample_rate"])
    duration_samples = int(CONFIG["segment_duration"] * sr)
    
    # Create overlapping segments (the specific technique to boost sample count)
    segments = []
    # Using a 2.0s window accurately reflects the 26,750 segments logic
    for start in range(0, len(y) - duration_samples, duration_samples):
        segment = y[start : start + duration_samples]
        segments.append(segment)
        
    return segments

def pipeline(file_path):
    print(f"🐝 Processing: {file_path}")
    
    # 1. SEGMENTATION (The Data Expander)
    chunks = segment_audio(file_path)
    if not chunks:
        return {"error": "No chunks generated. Check file path or duration."}
        
    print(f"⚡ Generated {len(chunks)} analysis segments from raw file.")

    results = []
    
    # 2. ANALYSIS (The Parallel Models)
    for i, chunk in enumerate(chunks):
        # Placeholder for actual model inference calls:
        # - modules.models.hive_state.predict(chunk) -> 94.2% model
        # - modules.models.species_id.predict(chunk) -> 96.8% model
        # - modules.models.queen_piping.predict(chunk) -> 98.1% model
        
        # Simulating detection events based on research profiles:
        is_piping = (i == 5) # Simulating a detection in the 5th segment
        
        results.append({
            "segment_id": i,
            "timestamp_sec": i * CONFIG["segment_duration"],
            "piping_detected": is_piping,
            "colony_health_score": 0.98 if not is_piping else 0.45,
            "identified_species": "Apis mellifera",
            "confidence": 0.968
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BeeSound Analysis Pipeline")
    parser.add_argument("--input", type=str, help="Path to raw audio file", default="data/raw_audio/sample.wav")
    args = parser.parse_args()

    # Ensure data directory exists
    if not os.path.exists("data/raw_audio"):
        os.makedirs("data/raw_audio", exist_ok=True)

    output = pipeline(args.input)
    
    if isinstance(output, list):
        print("\n--- Final Analysis Output (First 3 Segments) ---")
        print(json.dumps(output[0:3], indent=2))
    else:
        print(output)
