"""
BEESOUND ANALYSIS - Health State Classifier
Stage 2: Is the Hive Healthy?
Model: CNN trained on MFCC features
Target Accuracy: 94.2%

Inference modes:
  NEURAL     — Trained .pth weights loaded from weights/hive_state.pth
               (falls back to weights/beesound_final_v3.pth)
  HEURISTIC  — Spectral centroid / ZCR rules (always available)
"""

import os
import sys
import logging
import numpy as np
import librosa

logger = logging.getLogger(__name__)

_torch = None

def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


class HealthStateClassifier:
    """
    Classifier: Healthy vs. Queenless
    Based on Audio_based_identification_beehive_states repository.
    
    Automatically loads trained weights when available, otherwise falls
    back to acoustic heuristics.  Check ``self.inference_mode`` to see
    which path is active.
    """
    
    def __init__(self, weights_path=None, num_classes=2):
        self.classes = ['Healthy', 'Queenless', 'Swarming', 'Stressed']
        self.num_classes = num_classes
        self.model = None
        self.inference_mode = 'heuristic'

        # Resolve weights — try hive_state.pth first, then beesound_final_v3.pth
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            weights_path,
            os.path.join(project_root, 'weights', 'hive_state.pth'),
            os.path.join(project_root, 'weights', 'beesound_final_v3.pth'),
        ]

        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                self._try_load_weights(candidate)
                if self.inference_mode == 'neural':
                    break

        if self.inference_mode != 'neural':
            logger.info(
                "HealthStateClassifier: No valid weights found — using HEURISTIC mode"
            )

    def _try_load_weights(self, weights_path):
        """Attempt to load neural network weights."""
        try:
            torch = _get_torch()
            tools_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools'
            )
            if tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from train_architecture import BeeDeepArchitecture  # type: ignore

            model = BeeDeepArchitecture(num_classes=self.num_classes)
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()

            self.model = model
            self.inference_mode = 'neural'
            size_mb = os.path.getsize(weights_path) / (1024 * 1024)
            logger.info(
                "HealthStateClassifier: Loaded NEURAL weights (%.1f MB) from %s",
                size_mb, weights_path,
            )
        except Exception as e:
            logger.warning(
                "HealthStateClassifier: Failed to load weights from %s (%s)",
                weights_path, e,
            )

    def extract_features(self, audio, sample_rate=22050):
        """
        Extract MFCC features from audio segment.
        
        MFCCs (Mel-Frequency Cepstral Coefficients) capture the
        spectral envelope of the sound - the "timbre" of the hive.
        
        Args:
            audio: Audio segment (numpy array)
            sample_rate: Sample rate in Hz
            
        Returns:
            Feature vector (numpy array)
        """
        # Extract MFCCs (13 coefficients is standard)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        
        # Compute statistics across time
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
        
        # Concatenate features
        features = np.concatenate([mfcc_mean, mfcc_std, mfcc_delta])
        
        return features
    
    def predict(self, audio, sample_rate=22050):
        """
        Predict colony health state.
        
        Dispatches to neural or heuristic backend depending on
        whether trained weights were loaded.
        
        Args:
            audio: Audio segment
            sample_rate: Sample rate in Hz
            
        Returns:
            dict: {'state': str, 'confidence': float, 'probabilities': dict,
                   'inference_mode': str}
        """
        if self.inference_mode == 'neural' and self.model is not None:
            return self._predict_neural(audio, sample_rate)
        return self._predict_heuristic(audio, sample_rate)

    # ── Neural inference ───────────────────────────────────────────────────

    def _predict_neural(self, audio, sample_rate):
        """Run real CNN forward pass on mel-spectrogram."""
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
        confidence = float(probs[pred_idx])

        # Map to class label
        class_map = self.classes[:self.num_classes]
        state = class_map[pred_idx] if pred_idx < len(class_map) else f"class_{pred_idx}"

        probabilities = {cls: 0.0 for cls in self.classes}
        for i, p in enumerate(probs):
            if i < len(self.classes):
                probabilities[self.classes[i]] = float(p)

        return {
            'state': state,
            'confidence': confidence,
            'probabilities': probabilities,
            'inference_mode': 'neural',
        }

    # ── Heuristic fallback (original behavior) ─────────────────────────────

    def _predict_heuristic(self, audio, sample_rate):
        """
        Heuristic classification using spectral features.
        This is the original behavior — unchanged.
        """
        # Extract features
        features = self.extract_features(audio, sample_rate)
        
        # Analyze frequency distribution
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate))
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Simple heuristic classifier (replace with trained model)
        if spectral_centroid > 2000 and zero_crossing_rate > 0.1:
            state = 'Healthy'
            confidence = 0.942  # Target accuracy
        elif spectral_centroid < 1500:
            state = 'Queenless'
            confidence = 0.87
        elif zero_crossing_rate > 0.15:
            state = 'Swarming'
            confidence = 0.79
        else:
            state = 'Stressed'
            confidence = 0.72
        
        # Generate probability distribution
        probabilities = {cls: 0.0 for cls in self.classes}
        probabilities[state] = confidence
        remaining = 1.0 - confidence
        for cls in self.classes:
            if cls != state:
                probabilities[cls] = remaining / (len(self.classes) - 1)
        
        return {
            'state': state,
            'confidence': confidence,
            'probabilities': probabilities,
            'inference_mode': 'heuristic',
        }

if __name__ == "__main__":
    print("🩺 Health State Classifier")
    print("   Target: 94.2% accuracy (Nduva et al., 2023)")
    print("   Classes: Healthy, Queenless, Swarming, Stressed")
    hsc = HealthStateClassifier()
    print(f"   Mode:   {hsc.inference_mode.upper()}")
