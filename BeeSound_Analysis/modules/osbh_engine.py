"""
OSBH ENGINE WRAPPER
Integrates the Open Source Beehives (OSBH) / AudioHealth logic into the BeeSound ecosystem.
This module provides a pure-python implementation of the frequency-based heuristics
defined in the Hiveeyes/AudioHealth project.
"""

import numpy as np
from scipy import signal
import scipy.io.wavfile as wav

class OSBHEngine:
    def __init__(self):
        # OSBH Frequencies of Interest (in Hz)
        self.frequencies = {
            'active_band': (220, 275),
            'alert_band': (445, 525),
            'piping_peak': 450,
            'swarming_range': (200, 300)
        }
        
    def analyze_audio(self, wav_path):
        """
        Implements the AudioHealth 'power' analysis logic.
        """
        try:
            fs, x = wav.read(wav_path)
            # Standardize to mono
            if x.ndim == 2:
                x = x[:, 0]
            
            # Welch's power spectral density
            f, Pxx_spec = signal.welch(x, fs, 'flattop', 1024, scaling='spectrum')
            rms_spec = np.sqrt(Pxx_spec)
            
            # Peak detection (linear spectrum)
            peak_indices = signal.argrelmax(rms_spec)[0]
            peak_freqs = f[peak_indices]
            peak_powers = rms_spec[peak_indices]
            
            peak_data = dict(zip(peak_freqs, peak_powers))
            
            # Application of OSBH Analysis Rules (Strategy lr-2.1)
            results = self._apply_osbh_rules(peak_data)
            return results
        except Exception as e:
            return {"error": str(e)}

    def _apply_osbh_rules(self, peak_data):
        """
        Pure Python port of the AudioHealth decision logic.
        """
        # 1. Activity Check (~250Hz)
        band250 = {freq: pwr for freq, pwr in peak_data.items() if 220 <= freq <= 275}
        power250 = max(band250.values()) if band250 else 0
        
        # 2. Alert Check (~500Hz)
        band500 = {freq: pwr for freq, pwr in peak_data.items() if 445 <= freq <= 525}
        power500 = max(band500.values()) if band500 else 0
        
        # 3. Ratio Calculation
        ratio = float(power500) / float(power250) if power250 > 0 else 0
        
        # OSBH Thresholds
        is_healthy = power250 >= 1000
        is_queenless = ratio >= 0.6
        
        state = "ACTIVE" if is_healthy else "LOW_ACTIVITY"
        if is_queenless:
            state = "QUEEN_MISSING"
            
        return {
            "state": state,
            "activity_rms": power250,
            "alert_rms": power500,
            "ratio": ratio,
            "threshold_met": is_queenless
        }

if __name__ == "__main__":
    # Test stub
    engine = OSBHEngine()
    print("OSBH Engine Initialized and Aligned.")
