"""
BEESOUND ANALYSIS - Event Detector
Stage 3: Is there an Emergency?
Model: Frequency-domain pattern matching for queen piping
Target Recall: 98.1%

Inference modes:
  NEURAL     — Trained .pth weights loaded from weights/event_detector.pth
  HEURISTIC  — scipy signal processing (always available, already strong for piping)
"""

import os
import sys
import logging
import numpy as np
import librosa
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class EventDetector:
    """
    Detector: Queen Piping & Swarming Signals
    Based on beepiping repository (Fourer & Orlorwska, DCASE 2022).
    
    Automatically loads trained weights when available, otherwise falls
    back to DSP heuristics (which are already strong for piping detection).
    Check ``self.inference_mode`` to see which path is active.
    """
    
    def __init__(self, weights_path=None, num_classes=2):
        # Queen piping characteristics
        self.piping_freq_range = (300, 500)  # Hz
        self.piping_duration_min = 0.1       # seconds
        self.piping_threshold = 0.7          # confidence threshold
        self.num_classes = num_classes
        self.model = None
        self.inference_mode = 'heuristic'

        # Resolve weights path
        if weights_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            weights_path = os.path.join(project_root, 'weights', 'event_detector.pth')

        self._try_load_weights(weights_path)

    def _try_load_weights(self, weights_path):
        """Attempt to load neural network weights."""
        if not os.path.isfile(weights_path):
            logger.info(
                "EventDetector: No weights at %s — using HEURISTIC mode",
                weights_path,
            )
            return

        try:
            torch = _get_torch()
            tools_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
            )
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from train_architecture import BeeDeepArchitecture

            model = BeeDeepArchitecture(num_classes=self.num_classes)
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            self.inference_mode = 'neural'
            size_mb = os.path.getsize(weights_path) / (1024 * 1024)
            logger.info(
                "EventDetector: Loaded NEURAL weights (%.1f MB) from %s",
                size_mb, weights_path,
            )
        except Exception as e:
            logger.warning(
                "EventDetector: Failed to load weights (%s) — falling back to HEURISTIC",
                e,
            )
            self.model = None
            self.inference_mode = 'heuristic'

    def detect_piping(self, audio, sample_rate=22050):
        """
        Detect queen piping signals.
        
        Queen piping is a high-pitched "toot" sound (300-500 Hz)
        that indicates:
        - Multiple queens in hive (pre-swarm)
        - Queen emergence imminent
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: {'detected': bool, 'confidence': float, 'timestamps': list}
        """
        # Compute spectrogram
        f, t, Sxx = scipy_signal.spectrogram(audio, sample_rate, nperseg=1024)
        
        # Find frequency bins corresponding to piping range
        freq_mask = (f >= self.piping_freq_range[0]) & (f <= self.piping_freq_range[1])
        piping_band = Sxx[freq_mask, :]
        
        # Compute energy in piping frequency band
        piping_energy = np.sum(piping_band, axis=0)
        
        # Normalize
        if np.max(piping_energy) > 0:
            piping_energy = piping_energy / np.max(piping_energy)
        
        # Detect peaks (potential piping events)
        threshold = self.piping_threshold
        peaks, properties = scipy_signal.find_peaks(
            piping_energy,
            height=threshold,
            distance=int(self.piping_duration_min * len(t) / t[-1])
        )
        
        # Convert peak indices to timestamps
        timestamps = [t[peak] for peak in peaks]
        
        # Calculate confidence based on peak strength
        if len(peaks) > 0:
            confidence = float(np.mean(properties['peak_heights']))
            detected = True
        else:
            confidence = 0.0
            detected = False
        
        return {
            'detected': detected,
            'confidence': confidence,
            'event_count': len(peaks),
            'timestamps': timestamps,
            'frequency_range': self.piping_freq_range
        }
    
    def detect_hissing(self, audio, sample_rate=22050):
        """
        Detect defensive hissing (high-frequency broadband noise).
        
        Indicates:
        - Hive disturbance
        - Defensive behavior
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: {'detected': bool, 'confidence': float}
        """
        # Compute spectral flatness (measure of noise-like quality)
        flatness = librosa.feature.spectral_flatness(y=audio)
        
        # High flatness = noise-like = potential hissing
        mean_flatness = np.mean(flatness)
        
        # Hissing is typically > 0.5 flatness
        detected = mean_flatness > 0.5
        confidence = float(min(mean_flatness / 0.5, 1.0))
        
        return {
            'detected': detected,
            'confidence': confidence,
            'flatness': float(mean_flatness)
        }
    
    def analyze(self, audio, sample_rate=22050):
        """
        Run full event detection pipeline.
        
        Dispatches to neural or heuristic backend depending on
        whether trained weights were loaded.
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: Combined results from all detectors
        """
        if self.inference_mode == 'neural' and self.model is not None:
            return self._analyze_neural(audio, sample_rate)
        return self._analyze_heuristic(audio, sample_rate)

    # ── Neural inference ───────────────────────────────────────────────────

    def _analyze_neural(self, audio, sample_rate):
        """Run CNN-based event classification + DSP piping detection."""
        torch = _get_torch()

        mel = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, fmax=8000
        )
        mel_db = (librosa.power_to_db(mel, ref=np.max) + 40) / 40

        target_width = 87
        if mel_db.shape[1] < target_width:
            mel_db = np.pad(mel_db, ((0, 0), (0, target_width - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :target_width]

        tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().numpy()

        pred_idx = int(np.argmax(probs))
        neural_confidence = float(probs[pred_idx])
        neural_event = pred_idx == 1  # class 1 = event detected

        # Also run DSP piping detection (always useful)
        piping = self.detect_piping(audio, sample_rate)
        hissing = self.detect_hissing(audio, sample_rate)

        # Combined decision: neural OR DSP
        piping_detected = piping['detected'] or neural_event

        if piping_detected:
            alert_level = 'CRITICAL'
            alert_message = 'Queen piping detected - Swarm imminent!'
        elif hissing['detected']:
            alert_level = 'WARNING'
            alert_message = 'Defensive behavior detected'
        else:
            alert_level = 'NORMAL'
            alert_message = 'No emergency signals detected'

        return {
            'alert_level': alert_level,
            'alert_message': alert_message,
            'piping': piping,
            'hissing': hissing,
            'neural_confidence': neural_confidence,
            'inference_mode': 'neural',
        }

    # ── Heuristic fallback (original behavior) ─────────────────────────────

    def _analyze_heuristic(self, audio, sample_rate):
        """
        DSP-based event detection.
        This is the original behavior — unchanged.
        """
        piping = self.detect_piping(audio, sample_rate)
        hissing = self.detect_hissing(audio, sample_rate)
        
        # Determine overall alert level
        if piping['detected']:
            alert_level = 'CRITICAL'
            alert_message = 'Queen piping detected - Swarm imminent!'
        elif hissing['detected']:
            alert_level = 'WARNING'
            alert_message = 'Defensive behavior detected'
        else:
            alert_level = 'NORMAL'
            alert_message = 'No emergency signals detected'
        
        return {
            'alert_level': alert_level,
            'alert_message': alert_message,
            'piping': piping,
            'hissing': hissing,
            'inference_mode': 'heuristic',
        }

if __name__ == "__main__":
    print("🚨 Event Detector")
    print("   Target: 98.1% recall for piping detection")
    print("   Signals: Queen piping (300-500Hz), Defensive hissing")
    ed = EventDetector()
    print(f"   Mode:   {ed.inference_mode.upper()}")
