# BeeSound Weights Directory

This is the **canonical location** for all trained model checkpoints.
The analysis pipeline (`models/*.py`) automatically loads weights from here
when they exist, falling back to heuristic classification when they don't.

## Expected Files

| Filename | Format | Producer | Used By |
|---|---|---|---|
| `beesound_final_v3.pth` | PyTorch state_dict | `tools/train_architecture.py` | `HealthStateClassifier` (fallback) |
| `bee_brain_v3.onnx` | ONNX | `tools/export_brain.py` | Cross-platform inference |
| `bee_brain_v3_int8.tflite` | TFLite Int8 | `tools/quantize_brain.py` | ESP32 edge deployment |
| `species_id.pth` | PyTorch state_dict | Custom / `import_checkpoint.py` | `SpeciesIdentifier` |
| `hive_state.pth` | PyTorch state_dict | Custom / `import_checkpoint.py` | `HealthStateClassifier` |
| `event_detector.pth` | PyTorch state_dict | Custom / `import_checkpoint.py` | `EventDetector` |

## How to Populate

### Option A: Train from scratch
```bash
# Requires a labeled dataset manifest (CSV with file_path, label columns)
python tools/train_architecture.py
# Output: beesound_final_v3.pth (in CWD — move to weights/ after)
```

### Option B: Import from external training (Kaggle, Colab, etc.)
```bash
python tools/import_checkpoint.py --source /path/to/model.pth --slot hive_state
python tools/import_checkpoint.py --source /path/to/model.pth --slot species_id
python tools/import_checkpoint.py --source /path/to/model.pth --slot event_detector
```

### Option C: Download upstream pretrained weights (species_id submodule)
```bash
cd modules/models/species_id
bash download_pretrained_weights.sh
```

## Checking Status

```bash
python tools/run_analysis.py --health
# or
python -m models.model_inventory
```

## Architecture Compatibility

All `.pth` files are expected to be `state_dict` saves from `BeeDeepArchitecture`
(defined in `tools/train_architecture.py`). The architecture is a ResNet-style CNN:

- Input: `(batch, 1, 128, 87)` — mel-spectrogram
- Output: `(batch, num_classes)` — class logits
- Default: `num_classes=2` (healthy vs. queenless)

If you trained with a different `num_classes`, pass it via the model constructor.
