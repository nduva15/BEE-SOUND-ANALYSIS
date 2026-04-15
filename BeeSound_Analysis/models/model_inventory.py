"""
BEESOUND ANALYSIS - Model Inventory
Scans for trained model weights and reports deployment readiness.

Usage:
    python -m models.model_inventory          # Print status badge
    python models/model_inventory.py          # Same thing, standalone
"""

import os
import sys
import json
import glob
from datetime import datetime


def _ensure_utf8_stdout():
    """Reconfigure stdout for UTF-8 on Windows (avoids cp1252 crashes)."""
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass


# ── Known Model Slots ──────────────────────────────────────────────────────
# Each slot defines: human label, expected path(s) relative to project root,
# the training script that produces it, and the format.

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SLOTS = {
    # ── DeepBrain v3.1 (primary trained model) ──
    "beesound_v3": {
        "label": "DeepBrain v3.1 (PyTorch)",
        "paths": ["weights/beesound_final_v3.pth"],
        "producer": "tools/train_architecture.py",
        "format": ".pth",
    },
    "beesound_v3_onnx": {
        "label": "DeepBrain v3.1 (ONNX)",
        "paths": ["weights/bee_brain_v3.onnx"],
        "producer": "tools/export_brain.py",
        "format": ".onnx",
    },
    "beesound_v3_tflite": {
        "label": "DeepBrain v3.1 (TFLite Int8)",
        "paths": ["weights/bee_brain_v3_int8.tflite"],
        "producer": "tools/quantize_brain.py",
        "format": ".tflite",
    },

    # ── Per-stage custom weights (loaded by models/*.py) ──
    "species_id": {
        "label": "Species Identifier",
        "paths": ["weights/species_id.pth"],
        "producer": "Custom training / import_checkpoint.py",
        "format": ".pth",
    },
    "hive_state": {
        "label": "Health State Classifier",
        "paths": [
            "weights/hive_state.pth",
            "weights/beesound_final_v3.pth",  # fallback: DeepBrain works here too
        ],
        "producer": "tools/train_architecture.py / import_checkpoint.py",
        "format": ".pth",
    },
    "event_detector": {
        "label": "Event Detector (Piping/Hissing)",
        "paths": ["weights/event_detector.pth"],
        "producer": "Custom training / import_checkpoint.py",
        "format": ".pth",
    },

    # ── Upstream pretrained (species_id submodule) ──
    "species_panns": {
        "label": "PANNs (Cnn14) Pretrained",
        "paths": [
            "modules/models/species_id/models/cnns/panns/pretrained_weights/*.pth",
        ],
        "producer": "modules/models/species_id/download_pretrained_weights.sh",
        "format": ".pth",
    },
    "species_ast": {
        "label": "AST Pretrained",
        "paths": [
            "modules/models/species_id/models/transformers/ast/pretrained_weights/*.pth",
        ],
        "producer": "modules/models/species_id/download_pretrained_weights.sh",
        "format": ".pth",
    },
    "species_ssast": {
        "label": "SSAST Pretrained",
        "paths": [
            "modules/models/species_id/models/transformers/ssast/pretrained_weights/*.pth",
        ],
        "producer": "modules/models/species_id/download_pretrained_weights.sh",
        "format": ".pth",
    },
    "species_mae_ast": {
        "label": "MAE-AST Pretrained",
        "paths": [
            "modules/models/species_id/models/transformers/mae_ast/pretrained_weights/*.pt",
        ],
        "producer": "modules/models/species_id/download_pretrained_weights.sh",
        "format": ".pt",
    },
}


class ModelInventory:
    """
    Scans the project tree for trained model weights and reports status.
    """

    # Status codes
    LOADED  = "LOADED"
    MISSING = "MISSING"
    CORRUPT = "CORRUPT"

    def __init__(self, project_root=None):
        self.root = project_root or _PROJECT_ROOT
        self._cache = None

    # ── Core scan ──────────────────────────────────────────────────────────

    def scan(self, force=False):
        """
        Scan all known weight slots and return status dict.

        Returns:
            dict: {slot_name: {label, status, path, size_mb, producer, format}}
        """
        if self._cache and not force:
            return self._cache

        results = {}
        for slot_name, spec in SLOTS.items():
            found_path = None
            file_size = 0

            for pattern in spec["paths"]:
                full_pattern = os.path.join(self.root, pattern)
                # Support glob patterns (for pretrained_weights dirs)
                matches = glob.glob(full_pattern)
                if matches:
                    found_path = matches[0]
                    break

            if found_path and os.path.isfile(found_path):
                file_size = os.path.getsize(found_path)
                # Basic corruption check: file must be > 0 bytes
                if file_size > 0:
                    status = self.LOADED
                else:
                    status = self.CORRUPT
            else:
                status = self.MISSING

            results[slot_name] = {
                "label": spec["label"],
                "status": status,
                "path": found_path,
                "size_mb": round(file_size / (1024 * 1024), 2) if file_size else 0,
                "producer": spec["producer"],
                "format": spec["format"],
            }

        self._cache = results
        return results

    # ── Convenience accessors ──────────────────────────────────────────────

    def find_weight(self, slot_name):
        """
        Return the resolved file path for a slot, or None if missing/corrupt.
        """
        inv = self.scan()
        entry = inv.get(slot_name)
        if entry and entry["status"] == self.LOADED:
            return entry["path"]
        return None

    def summary_line(self):
        """
        One-line summary for embedding in CLI output.
        Example: "species_id=HEURISTIC | health=HEURISTIC | events=HEURISTIC (0/3 neural)"
        """
        inv = self.scan()
        stages = [
            ("species_id", "species_id"),
            ("health", "hive_state"),
            ("events", "event_detector"),
        ]
        parts = []
        neural_count = 0
        for display, slot in stages:
            if inv.get(slot, {}).get("status") == self.LOADED:
                parts.append(f"{display}=NEURAL")
                neural_count += 1
            else:
                parts.append(f"{display}=HEURISTIC")

        return f"{' | '.join(parts)} ({neural_count}/3 neural)"

    # ── Display ────────────────────────────────────────────────────────────

    def print_badge(self):
        """
        Print a rich ASCII status table to the terminal.
        """
        _ensure_utf8_stdout()
        inv = self.scan(force=True)

        # Status symbols
        icons = {
            self.LOADED:  "✅",
            self.MISSING: "❌",
            self.CORRUPT: "⚠️ ",
        }

        loaded = sum(1 for v in inv.values() if v["status"] == self.LOADED)
        total  = len(inv)

        print()
        print("╔══════════════════════════════════════════════════════════════════╗")
        print("║          🧠  BEESOUND MODEL INVENTORY                          ║")
        print("╠══════════════════════════════════════════════════════════════════╣")

        # Group: Core inference models
        print("║                                                                  ║")
        print("║  ── Core Inference Models ──                                     ║")
        for slot in ["species_id", "hive_state", "event_detector"]:
            entry = inv[slot]
            icon = icons[entry["status"]]
            size = f"({entry['size_mb']} MB)" if entry["status"] == self.LOADED else ""
            line = f"  {icon} {entry['label']:<35} {entry['status']:<10} {size}"
            print(f"║{line:<66}║")

        # Group: DeepBrain export chain
        print("║                                                                  ║")
        print("║  ── DeepBrain v3.1 Export Chain ──                               ║")
        for slot in ["beesound_v3", "beesound_v3_onnx", "beesound_v3_tflite"]:
            entry = inv[slot]
            icon = icons[entry["status"]]
            size = f"({entry['size_mb']} MB)" if entry["status"] == self.LOADED else ""
            line = f"  {icon} {entry['label']:<35} {entry['status']:<10} {size}"
            print(f"║{line:<66}║")

        # Group: Upstream pretrained
        print("║                                                                  ║")
        print("║  ── Upstream Pretrained (species_id submodule) ──                ║")
        for slot in ["species_panns", "species_ast", "species_ssast", "species_mae_ast"]:
            entry = inv[slot]
            icon = icons[entry["status"]]
            size = f"({entry['size_mb']} MB)" if entry["status"] == self.LOADED else ""
            line = f"  {icon} {entry['label']:<35} {entry['status']:<10} {size}"
            print(f"║{line:<66}║")

        print("║                                                                  ║")
        print("╠══════════════════════════════════════════════════════════════════╣")
        summary = f"  Total: {loaded}/{total} models loaded"
        if loaded == 0:
            summary += "  ⚡ All stages running on HEURISTIC fallback"
        elif loaded == total:
            summary += "  🚀 Full neural deployment!"
        print(f"║{summary:<66}║")
        print("╚══════════════════════════════════════════════════════════════════╝")
        print()

    def to_json(self):
        """
        Return inventory as JSON-serializable dict (for API/health endpoints).
        """
        inv = self.scan(force=True)
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "project_root": self.root,
            "models": inv,
            "summary": {
                "total": len(inv),
                "loaded": sum(1 for v in inv.values() if v["status"] == self.LOADED),
                "missing": sum(1 for v in inv.values() if v["status"] == self.MISSING),
                "corrupt": sum(1 for v in inv.values() if v["status"] == self.CORRUPT),
            },
        }


# ── Standalone entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    inventory = ModelInventory()
    inventory.print_badge()

    print("📋 JSON Output:")
    print(json.dumps(inventory.to_json(), indent=2))
