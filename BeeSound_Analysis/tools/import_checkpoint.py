"""
BEESOUND ANALYSIS - Checkpoint Import Tool
Import externally trained model weights into the canonical weights/ directory.

Usage:
    python tools/import_checkpoint.py --source path/to/model.pth --slot hive_state
    python tools/import_checkpoint.py --source path/to/model.pth --slot species_id
    python tools/import_checkpoint.py --source path/to/model.pth --slot event_detector
    python tools/import_checkpoint.py --source path/to/model.pth --slot beesound_v3
"""

import os
import sys
import shutil
import argparse

# Add parent dir for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Slot → filename mapping ───────────────────────────────────────────────

SLOT_MAP = {
    'beesound_v3': {
        'filename': 'beesound_final_v3.pth',
        'description': 'DeepBrain v3.1 (primary trained model)',
        'num_classes': 2,
    },
    'species_id': {
        'filename': 'species_id.pth',
        'description': 'Species Identifier (6-class)',
        'num_classes': 6,
    },
    'hive_state': {
        'filename': 'hive_state.pth',
        'description': 'Health State Classifier (2-class)',
        'num_classes': 2,
    },
    'event_detector': {
        'filename': 'event_detector.pth',
        'description': 'Event Detector — piping/normal (2-class)',
        'num_classes': 2,
    },
}


def validate_checkpoint(source_path, num_classes):
    """
    Validate that a checkpoint can be loaded by BeeDeepArchitecture.
    Returns (success: bool, message: str, param_count: int).
    """
    try:
        import torch
        from tools.train_architecture import BeeDeepArchitecture

        # Load state dict
        state_dict = torch.load(source_path, map_location='cpu', weights_only=True)

        # Try instantiating the model and loading
        model = BeeDeepArchitecture(num_classes=num_classes)
        model.load_state_dict(state_dict)
        model.eval()

        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())

        # Smoke test: forward pass with dummy data
        dummy_input = torch.randn(1, 1, 128, 87)
        with torch.no_grad():
            output = model(dummy_input)

        expected_shape = (1, num_classes)
        if output.shape != torch.Size(expected_shape):
            return False, f"Output shape mismatch: expected {expected_shape}, got {tuple(output.shape)}", 0

        return True, "Checkpoint validated successfully", param_count

    except Exception as e:
        return False, f"Validation failed: {e}", 0


def import_checkpoint(source_path, slot_name, force=False, skip_validation=False):
    """
    Import a checkpoint into the canonical weights/ directory.
    """
    if slot_name not in SLOT_MAP:
        print(f"❌ Unknown slot '{slot_name}'")
        print(f"   Available slots: {', '.join(SLOT_MAP.keys())}")
        return False

    slot = SLOT_MAP[slot_name]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weights_dir = os.path.join(project_root, 'weights')
    dest_path = os.path.join(weights_dir, slot['filename'])

    print("=" * 60)
    print("🔧 BEESOUND CHECKPOINT IMPORT")
    print("=" * 60)
    print(f"   Source:      {source_path}")
    print(f"   Slot:        {slot_name} ({slot['description']})")
    print(f"   Destination: {dest_path}")
    print(f"   Classes:     {slot['num_classes']}")
    print()

    # Check source exists
    if not os.path.isfile(source_path):
        print(f"❌ Source file not found: {source_path}")
        return False

    source_size = os.path.getsize(source_path)
    print(f"📏 Source size: {source_size / (1024 * 1024):.2f} MB")

    # Check destination
    if os.path.exists(dest_path) and not force:
        print(f"⚠️  Destination already exists: {dest_path}")
        print(f"   Use --force to overwrite.")
        return False

    # Validate
    if not skip_validation:
        print(f"\n🔬 Validating checkpoint against BeeDeepArchitecture(num_classes={slot['num_classes']})...")
        success, message, param_count = validate_checkpoint(source_path, slot['num_classes'])

        if success:
            print(f"   ✅ {message}")
            print(f"   📊 Parameters: {param_count:,}")
        else:
            print(f"   ❌ {message}")
            print(f"   The checkpoint may be incompatible with this architecture.")
            print(f"   Use --skip-validation to bypass this check.")
            return False
    else:
        print(f"\n⚠️  Skipping validation (--skip-validation)")

    # Copy
    print(f"\n📦 Copying to {dest_path}...")
    os.makedirs(weights_dir, exist_ok=True)
    shutil.copy2(source_path, dest_path)

    dest_size = os.path.getsize(dest_path)
    print(f"✅ Import successful!")
    print(f"   File size: {dest_size / (1024 * 1024):.2f} MB")

    # Verify inventory
    print(f"\n📋 Updated inventory status:")
    from models.model_inventory import ModelInventory
    inventory = ModelInventory(project_root)
    inv = inventory.scan(force=True)
    entry = inv.get(slot_name, {})
    print(f"   {slot_name}: {entry.get('status', 'UNKNOWN')}")

    print(f"\n💡 The model will auto-load on next analysis run.")
    print(f"   Try: python tools/run_analysis.py --input your_audio.wav")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Import trained model checkpoints into BeeSound Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/import_checkpoint.py --source model.pth --slot hive_state
  python tools/import_checkpoint.py --source model.pth --slot species_id --force
  python tools/import_checkpoint.py --source model.pth --slot beesound_v3

Available slots:
  beesound_v3     DeepBrain v3.1 (primary, 2-class)
  species_id      Species Identifier (6-class)
  hive_state      Health State Classifier (2-class)
  event_detector  Event Detector — piping/normal (2-class)
        """
    )
    parser.add_argument('--source', type=str, required=True,
                        help='Path to the .pth checkpoint file')
    parser.add_argument('--slot', type=str, required=True,
                        choices=list(SLOT_MAP.keys()),
                        help='Target model slot')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing weights')
    parser.add_argument('--skip-validation', action='store_true',
                        help='Skip architecture validation (use with caution)')

    args = parser.parse_args()
    success = import_checkpoint(args.source, args.slot, args.force, args.skip_validation)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
