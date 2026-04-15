"""
BEESOUND ANALYSIS - Master Execution Script
The Complete 3-Stage Analysis Pipeline

Usage:
    python tools/run_analysis.py --input data/raw_audio/recording.wav
    python tools/run_analysis.py --health          # Show model inventory
    python tools/run_analysis.py --inventory-json   # JSON inventory output
"""

import sys
import os
import argparse
import librosa
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import AudioSegmenter, AudioCleaner, SpectrogramVisualizer
from models import SpeciesIdentifier, HealthStateClassifier, EventDetector
from models.model_inventory import ModelInventory


def print_header():
    """Print analysis header."""
    print("=" * 70)
    print("🐝 BEESOUND ANALYSIS PIPELINE")
    print("=" * 70)
    print()


def print_stage(stage_num, stage_name):
    """Print stage header."""
    print(f"\n{'─' * 70}")
    print(f"STAGE {stage_num}: {stage_name}")
    print(f"{'─' * 70}")


def print_model_status(species_model, health_model, event_model):
    """Print compact one-line model status summary."""
    modes = {
        'species_id': species_model.inference_mode,
        'health': health_model.inference_mode,
        'events': event_model.inference_mode,
    }
    neural_count = sum(1 for m in modes.values() if m == 'neural')

    parts = [f"{k}={v.upper()}" for k, v in modes.items()]
    line = " | ".join(parts)
    print(f"🧠 Models: {line} ({neural_count}/3 neural)")


def analyze_audio(file_path=None, audio_data=None, sample_rate=22050, save_spectrogram=True, verbose=True):
    """
    Run complete analysis pipeline on audio.
    Supports both file path (local) and raw audio data (cloud/RAM).
    
    Args:
        file_path: Path to audio file (optional)
        audio_data: Numpy array of audio samples (optional)
        sample_rate: Sample rate of audio_data (default: 22050)
        save_spectrogram: Whether to save spectrogram visualization
        verbose: Whether to print detailed output
        
    Returns:
        dict: Complete analysis results
    """
    # 1. Load Audio
    y = None
    sr = sample_rate
    duration = 0

    if audio_data is not None:
        # Case A: In-Memory Data (Kaggle/Cloud)
        y = audio_data
        duration = len(y) / sr
        file_label = "Raw Data Stream"
        if verbose:
            print_header()
            print(f"📊 Analyzing Raw Data: {duration:.1f}s @ {sr}Hz")

    elif file_path:
        # Case B: File on Disk
        if not os.path.exists(file_path):
            print(f"❌ Error: File not found: {file_path}")
            return None
        
        file_label = file_path
        if verbose:
            print_header()
            print(f"📂 Input File: {file_path}")
            print(f"📊 File Size: {os.path.getsize(file_path) / 1024:.1f} KB")
        
        if verbose: print(f"\n⏳ Loading audio...")
        y, sr = librosa.load(file_path, sr=22050)
        duration = len(y) / sr
        if verbose: print(f"✅ Loaded: {duration:.1f} seconds @ {sr} Hz")
    else:
        print("❌ Error: Must provide either file_path or audio_data")
        return None

    # 2. Initialize Components
    # Note: segmenter now accepts audio_data directly!
    segmenter = AudioSegmenter(window_size=2.0, overlap=0.5, sample_rate=sr)
    cleaner = AudioCleaner(sample_rate=sr)
    visualizer = SpectrogramVisualizer(sample_rate=sr)
    
    species_model = SpeciesIdentifier()
    health_model = HealthStateClassifier()
    event_model = EventDetector()

    # Print model status
    if verbose:
        print()
        print_model_status(species_model, health_model, event_model)
    
    # 3. Segment Audio
    if verbose: print(f"\n🔪 Segmenting audio into 2-second windows...")
    
    # Use the refactored segmenter which handles both file/data
    segments, timestamps = segmenter.segment_audio(file_path=file_path, audio_data=y, sr=sr)
    
    if len(segments) == 0:
        if verbose: print("⚠️  Audio too short for analysis (< 2.0s)")
        return None

    if verbose: print(f"✅ Generated {len(segments)} segments")
    
    # 4. Clean Audio
    if verbose: print(f"\n🧹 Cleaning audio (noise reduction + bandpass filter)...")
    cleaned_segments = [cleaner.clean(seg) for seg in segments]
    if verbose: print(f"✅ Cleaned {len(cleaned_segments)} segments")
    
    # 5. Generate Spectrogram (Only for files currently, to look nice)
    if save_spectrogram and file_path and verbose:
        print(f"\n📸 Generating spectrogram...")
        spec_path = visualizer.visualize_from_file(file_path)
        print(f"✅ Saved: {spec_path}")
    
    # 6. Analyze
    results = {
        'source': file_label,
        'duration': duration,
        'segment_count': len(segments)
    }
    
    # STAGE 1: Species
    if verbose: print_stage(1, "Species Identification (ViT Transformer)")
    species_votes = []
    species_mode = 'heuristic'
    for i, segment in enumerate(cleaned_segments[:5]):
        res = species_model.predict(segment, sr)
        species_votes.append(res['species'])
        species_mode = res.get('inference_mode', 'heuristic')
        if verbose and i==0:
            print(f"   Species: {res['species']} ({res['confidence']:.1%})")
            print(f"   Mode:    {species_mode.upper()}")
    
    most_common_species = max(set(species_votes), key=species_votes.count)
    results['species'] = {
        'identified': most_common_species,
        'inference_mode': species_mode,
    }
    
    # STAGE 2: Health
    if verbose: print_stage(2, "Colony Health Assessment (MFCC + CNN)")
    health_results = [health_model.predict(seg, sr) for seg in cleaned_segments]
    health_votes = [r['state'] for r in health_results]
    most_common_health = max(set(health_votes), key=health_votes.count)
    health_mode = health_results[0].get('inference_mode', 'heuristic') if health_results else 'heuristic'
    
    if verbose:
        print(f"   Colony State: {most_common_health}")
        print(f"   Mode:         {health_mode.upper()}")
    results['health'] = {
        'state': most_common_health,
        'inference_mode': health_mode,
    }
    
    # STAGE 3: Events
    if verbose: print_stage(3, "Emergency Signal Detection")
    piping = 0
    event_mode = 'heuristic'
    for seg in cleaned_segments:
        event_result = event_model.analyze(seg, sr)
        event_mode = event_result.get('inference_mode', 'heuristic')
        if event_result['piping']['detected']: piping += 1
            
    if verbose:
        if piping > 0: print(f"   🚨 ALERT: Queen piping detected in {piping} segments!")
        else: print(f"   ✅ No emergency signals detected")
        print(f"   Mode:    {event_mode.upper()}")

    results['events'] = {
        'piping_count': piping,
        'inference_mode': event_mode,
    }
    
    # Summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("📋 FINAL ANALYSIS SUMMARY")
        print(f"{'=' * 70}")
        print(f"   Species:      {results['species']['identified']}")
        print(f"   Health:       {results['health']['state']}")
        print(f"   Piping:       {piping > 0}") # Boolean
        print(f"   ─────────────────────────────────────")
        print(f"   Inference:    species={results['species']['inference_mode'].upper()}"
              f" | health={results['health']['inference_mode'].upper()}"
              f" | events={results['events']['inference_mode'].upper()}")
        print(f"{'=' * 70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='BEESOUND ANALYSIS')
    parser.add_argument('--input', type=str, help='Path to audio file')
    parser.add_argument('--no-spectrogram', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument(
        '--health', '--inventory',
        action='store_true',
        dest='show_inventory',
        help='Show model inventory status badge and exit',
    )
    parser.add_argument(
        '--inventory-json',
        action='store_true',
        help='Print model inventory as JSON and exit',
    )
    
    args = parser.parse_args()

    # ── Inventory modes ────────────────────────────────────────────────
    if args.show_inventory:
        inventory = ModelInventory()
        inventory.print_badge()
        return

    if args.inventory_json:
        import json
        inventory = ModelInventory()
        print(json.dumps(inventory.to_json(), indent=2))
        return

    # ── Analysis mode ──────────────────────────────────────────────────
    if not args.input:
        parser.error("--input is required for analysis (or use --health)")

    # Backward compatibility wrapper
    analyze_audio(
        file_path=args.input,
        save_spectrogram=not args.no_spectrogram,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main()
