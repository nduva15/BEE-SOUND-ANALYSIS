"""
BEESOUND ANALYSIS - Master Execution Script
The Complete 3-Stage Analysis Pipeline
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


def analyze_audio(file_path, save_spectrogram=True, verbose=True):
    """
    Run complete analysis pipeline on audio file.
    
    Args:
        file_path: Path to audio file
        save_spectrogram: Whether to save spectrogram visualization
        verbose: Whether to print detailed output
        
    Returns:
        dict: Complete analysis results
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return None
    
    if verbose:
        print_header()
        print(f"📂 Input File: {file_path}")
        print(f"📊 File Size: {os.path.getsize(file_path) / 1024:.1f} KB")
    
    # Load audio
    if verbose:
        print(f"\n⏳ Loading audio...")
    y, sr = librosa.load(file_path, sr=22050)
    duration = len(y) / sr
    if verbose:
        print(f"✅ Loaded: {duration:.1f} seconds @ {sr} Hz")
    
    # Initialize pipeline components
    segmenter = AudioSegmenter(window_size=2.0, overlap=0.5)
    cleaner = AudioCleaner(sample_rate=sr)
    visualizer = SpectrogramVisualizer(sample_rate=sr)
    
    # Initialize models
    species_model = SpeciesIdentifier()
    health_model = HealthStateClassifier()
    event_model = EventDetector()
    
    # Segment audio
    if verbose:
        print(f"\n🔪 Segmenting audio into 2-second windows...")
    segments, timestamps = segmenter.segment_audio(file_path)
    if verbose:
        print(f"✅ Generated {len(segments)} segments")
    
    # Clean audio
    if verbose:
        print(f"\n🧹 Cleaning audio (noise reduction + bandpass filter)...")
    cleaned_segments = [cleaner.clean(seg) for seg in segments]
    if verbose:
        print(f"✅ Cleaned {len(cleaned_segments)} segments")
    
    # Generate spectrogram
    if save_spectrogram and verbose:
        print(f"\n📸 Generating spectrogram...")
        spec_path = visualizer.visualize_from_file(file_path)
        print(f"✅ Saved: {spec_path}")
    
    # Analyze segments
    results = {
        'file': file_path,
        'duration': duration,
        'segment_count': len(segments),
        'segments': []
    }
    
    # STAGE 1: Species Identification
    if verbose:
        print_stage(1, "Species Identification (ViT Transformer)")
    
    species_votes = []
    for i, segment in enumerate(cleaned_segments[:5]):  # Analyze first 5 segments
        species_result = species_model.predict(segment, sr)
        species_votes.append(species_result['species'])
        if verbose and i == 0:
            print(f"   Species: {species_result['species']}")
            print(f"   Confidence: {species_result['confidence']:.1%}")
            print(f"   Is Bee: {species_result['is_bee']}")
    
    # Majority vote for species
    most_common_species = max(set(species_votes), key=species_votes.count)
    species_confidence = species_votes.count(most_common_species) / len(species_votes)
    
    results['species'] = {
        'identified': most_common_species,
        'confidence': species_confidence,
        'is_bee': 'Non-bee' not in most_common_species
    }
    
    # STAGE 2: Health State Classification
    if verbose:
        print_stage(2, "Colony Health Assessment (MFCC + CNN)")
    
    health_votes = []
    health_confidences = []
    for segment in cleaned_segments:
        health_result = health_model.predict(segment, sr)
        health_votes.append(health_result['state'])
        health_confidences.append(health_result['confidence'])
    
    # Aggregate health results
    most_common_health = max(set(health_votes), key=health_votes.count)
    avg_confidence = np.mean(health_confidences)
    
    if verbose:
        print(f"   Colony State: {most_common_health}")
        print(f"   Confidence: {avg_confidence:.1%}")
    
    results['health'] = {
        'state': most_common_health,
        'confidence': avg_confidence
    }
    
    # STAGE 3: Event Detection
    if verbose:
        print_stage(3, "Emergency Signal Detection (Frequency Analysis)")
    
    piping_detections = 0
    hissing_detections = 0
    all_events = []
    
    for i, segment in enumerate(cleaned_segments):
        event_result = event_model.analyze(segment, sr)
        all_events.append(event_result)
        
        if event_result['piping']['detected']:
            piping_detections += 1
        if event_result['hissing']['detected']:
            hissing_detections += 1
    
    if verbose:
        if piping_detections > 0:
            print(f"   🚨 ALERT: Queen piping detected in {piping_detections}/{len(segments)} segments!")
            print(f"   ⚠️  WARNING: Swarm imminent - immediate action required")
        elif hissing_detections > 0:
            print(f"   ⚠️  Defensive behavior detected in {hissing_detections}/{len(segments)} segments")
        else:
            print(f"   ✅ No emergency signals detected")
    
    results['events'] = {
        'piping_count': piping_detections,
        'hissing_count': hissing_detections,
        'alert_level': 'CRITICAL' if piping_detections > 0 else ('WARNING' if hissing_detections > 0 else 'NORMAL')
    }
    
    # Final Summary
    if verbose:
        print(f"\n{'=' * 70}")
        print("📋 FINAL ANALYSIS SUMMARY")
        print(f"{'=' * 70}")
        print(f"   Species:      {results['species']['identified']} ({results['species']['confidence']:.1%})")
        print(f"   Health:       {results['health']['state']} ({results['health']['confidence']:.1%})")
        print(f"   Alert Level:  {results['events']['alert_level']}")
        print(f"{'=' * 70}\n")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='BEESOUND ANALYSIS - Complete Acoustic Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input audio file'
    )
    parser.add_argument(
        '--no-spectrogram',
        action='store_true',
        help='Skip spectrogram generation'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    results = analyze_audio(
        args.input,
        save_spectrogram=not args.no_spectrogram,
        verbose=not args.quiet
    )
    
    if results is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
