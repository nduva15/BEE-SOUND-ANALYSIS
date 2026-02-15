# BEESOUND ANALYSIS 🐝🔊

## Project Mission
BeeSound Analysis is a unified open-source framework for assessing bee colony vitality through acoustic monitoring. We combine edge-computing firmware, bioacoustic signal processing, and transformer-based deep learning to decode the language of bees.

### 🛠 The Stack
* **Languages:** C++, Python 3.9+, R
* **Intelligence:** PyTorch, TensorFlow
* **Hardware:** Arduino / ESP32 (OSBH compatible)
* **Status:** 5 repos → 1 unified framework

---

## 🎙️ Source Repositories
The project merges five specialized research archives:

1. **`audiohealth` (C++ / Arduino)**
   * Role: Field Recorder / Edge Firmware
   * Real-time FFT on embedded hardware
   * Low-power audio capture from hive sensors
2. **`Audio_based_identification_beehive_states` (Python)**
   * Role: General Health Check
   * Classifies Queenless vs. Healthy colonies
   * MFCC and spectral feature extraction
3. **`Transformers-Bee-Species-Acoustic-Recognition` (Python / PyTorch)**
   * Role: Species Identifier
   * Vision Transformer (ViT) on spectrograms
   * Multi-species classification
4. **`beepiping` (Python)**
   * Role: Emergency Alarm
   * Queen piping frequency isolation
   * Swarming event prediction
5. **`bioacoustics` (R / C++)**
   * Role: Signal Pre-Processor
   * Automated noise reduction (wind, traffic)
   * Acoustic feature extraction pipeline

---

## 🏗 The Architecture
We operate in three distinct stages:

### 01. `source_capture` — The Ears
**Source:** Repo 1 (AudioHealth)
* **Location:** `modules/edge_firmware/`
* C++ firmware running on hive hardware. Captures raw audio and performs on-device FFT for initial spectral analysis.

### 02. `signal_processing` — The Translator
**Source:** Repo 5 (Bioacoustics)
* **Location:** `modules/feature_extractor/`
* R/C++ pre-processing pipeline. Removes environmental noise (wind, vehicles) and extracts acoustic features like MFCCs.

### 03. `brain_models` — The Intelligence
**Source:** Repos 2, 3 & 4
* **Location:** `modules/models/`
* Three-stage AI analysis: colony health classification, queen piping detection, and transformer-based species recognition.

---

## 📂 Folder Structure
```text
BeeSound_Analysis/
│
├── README.md                  <-- The Master Doc
├── requirements.txt            <-- Unified Python Dependencies
├── data/                      <-- UNIFIED Data Storage
│   ├── raw_audio/             ← .wav files from all sources
│   ├── processed_spectrograms/
│   └── datasets_metadata/     ← CSVs/Metadata describing the files
│
├── modules/
│   ├── edge_firmware/         ← [Repo 1] C++ device code
│   │   ├── src/
│   │   └── lib/
│   │
│   ├── feature_extractor/     ← [Repo 5] R/C++ audio cleaning
│   │   ├── noise_reduction/
│   │   └── extraction_scripts/
│   │
│   └── models/                ← [Repos 2, 3, 4] Python AI
│       ├── hive_state/        ← Queenless vs. Healthy
│       ├── species_id/        ← Transformers
│       └── queen_piping/      ← Event detection
│
└── tools/
    ├── pipeline.py            ← Orchestration script
    └── environment.yml        ← Unified Conda environment
```

---

## 📊 Research Findings & Key Results
| Metric | Result | Context |
| :--- | :--- | :--- |
| **Colony Health Accuracy** | **94.2%** | Nduva et al. (2023) |
| **Species ID Accuracy** | **96.8%** | ViT-based Recognition (2024) |
| **Piping Detection Recall** | **98.1%** | Apidologie Journal (2023) |
| **Edge Inference Latency** | **<200ms** | ESP32 Real-time FFT |
| **Total Audio Collected** | **530+ hrs** | Combined Research Dataset |
| **Combined Samples** | **26,750+** | Across all Hive States |

### 📄 Publications
* **Audio-based Identification of Beehive States** (Nduva et al., 2023): 94.2% accuracy using MFCC/SVM.
* **Transformer-Based Bee Species Acoustic Recognition** (2024): 96.8% accuracy via Vision Transformers.
* **Queen Piping Detection via Temporal Signal Analysis** (2023): 98.1% recall for swarming indicator.
* **Open Source Beehive Audio Health Monitoring** (2022): Low-cost edge monitoring benchmarks.

---

## 💾 Unified Datasets
| Dataset | Samples | Format | Duration | Source |
| :--- | :--- | :--- | :--- | :--- |
| **OSBH Hive Audio** | 12,000+ | WAV 44.1kHz | ~200 hrs | Repo 1 |
| **Beehive State Labels** | 2,400 | WAV + CSV | ~80 hrs | Repo 2 |
| **Multi-Species Acoustic** | 8,500 | WAV 48kHz | ~150 hrs | Repo 3 |
| **Queen Piping Events** | 650 | WAV segments | ~12 hrs | Repo 4 |
| **Environmental Baselines**| 3,200 | WAV + metadata| ~90 hrs | Repo 5 |

---

## ⚙️ Integration Pipeline & Reproducibility
The `tools/pipeline.py` script implements the **2.0s audio segmentation** logic used in our research to expand the dataset to 26,750 samples. This ensures that even long field recordings are analyzed with the same window size as our training data, maintaining the **94.2% - 98.1% accuracy benchmarks**.

### 🛠 Environment Setup
To replicate the exact research environment:
1. **Conda (Recommended):** `conda env create -f tools/environment.yml`
2. **Pip:** `pip install -r requirements.txt`

The environment is pinned to specific versions (Python 3.9, PyTorch 2.0.1, TensorFlow 2.12.0) to ensure cross-model compatibility between the bioacoustics and transformer pipelines.

### 🚀 Usage
```bash
python tools/pipeline.py --input data/raw_audio/recording.wav
```

**Sample Output:**
```json
{
  "segment_id": 5,
  "timestamp_sec": 10.0,
  "piping_detected": true,
  "colony_health_score": 0.45,
  "identified_species": "Apis mellifera",
  "confidence": 0.968
}
```
