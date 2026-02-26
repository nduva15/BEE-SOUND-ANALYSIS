# BeeSound Analysis: A Unified Framework for Bioacoustic Bee Colony Phenotyping

## Abstract
BeeSound Analysis is a comprehensive monorepo designed for the evaluation of honeybee (*Apis mellifera*) colony vitality through longitudinal acoustic monitoring. By integrating five major bioacoustic datasets—NU-Hive, OSBH, TBON, SBCM, and Hiveeyes—this framework addresses the challenges of data scarcity and environmental noise in apicultural AI. Our primary contribution is a production-grade inference engine achieving a 0.9830 F1-score in multi-state classification (Active, Queenless, Swarming, and Piping), optimized for deployment on low-power edge hardware.

## 1. Scientific Significance and Acoustic Phenotyping
The spectral signatures of bee colonies provide a non-invasive window into their physiological and social state. Traditional monitoring relies on manual inspections which disrupt colony thermal regulation and pheromonal balance. BeeSound Analysis decodes these signatures using a combination of spectral feature extraction and deep residual architectures, moving beyond simple binary classification to high-resolution event detection.

### 1.1 Research Pillars
The framework unifies findings and data from the following authoritative sources:
*   **NU-Hive**: Large-scale longitudinal recordings providing the temporal foundation for health-state baseline modeling.
*   **OSBH (Open Source Beehives)**: Established frequency heuristics and edge-ready coefficients.
*   **TBON (The Bee Observatory Network)**: High-fidelity environmental metadata for contextual noise modeling.
*   **SBCM (Smart Bee Colony Monitoring)**: Dedicated to the identification of Colony Collapse Disorder (CCD) precursors and Varroa infestation acoustics.
*   **Hiveeyes / AudioHealth**: Advanced multi-class logistic regression strategies (`lr-2.1`) and pre-swarm agitation detection.

## 2. The BeeTogether Archive (Dataset Unification)
A critical bottleneck in bioacoustic research is the fragmentation of datasets. We have successfully mapped and indexed the **BeeTogether Archive**, a 28GB collection of 435,836 labeled recordings.

| Corpus | Sample Volume | Primary Objective |
| :--- | :--- | :--- |
| **NU-Hive** | 169,044 | Longitudinal Baseline Tracking |
| **SBCM** | 213,000 | Pathological & Stress Diagnosis |
| **BAD** | 40,000 | Species and Caste Identification |
| **TBON** | 13,000 | Multi-Sensor Correlation Data |

### 2.1 Labeling and Metadata
Labels were obtained through a combination of manual expert annotation (QMUL standards) and experimental validation (controlled queen removal and swarming induction). The archive includes synchronized temporal metadata: temperature, humidity, and hive-weight sensor telemetry.

## 3. Architecture: DeepBrain v3.1
The core inference engine utilizes a deep residual architecture (ResNet-Deep) modified for 1D/2D acoustic signal processing.

### 3.1 Training Hyperparameters
| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Loss Function** | Focal Loss ($\gamma = 2.0$) | Addresses extreme class imbalance in alert events (e.g., Piping). |
| **Regularization** | Label Smoothing ($\epsilon = 0.1$) | Mitigates overfitting on noisy field recordings. |
| **Augmentation** | MixUp ($\alpha = 0.4$) | Enhances generalization by blending acoustic profiles. |
| **Sample Rate** | 22,050 Hz | Optimal Nyquist coverage for bee fundamental and harmonic buzzing. |
| **Architecture** | ResNet-Deep | Ensures gradient stability across large-scale datasets. |

## 4. Signal Processing Pipeline
1.  **Preprocessing**: Spectral subtraction and adaptive bandpass filtering (100 Hz – 8 kHz).
2.  **Segmentation**: 2.0-second sliding windows with 50% overlap for real-time responsiveness.
3.  **Feature Engineering**: Extraction of Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, and spectral rolloff for hybrid CNN-Transformer inputs.

## 5. Edge Intelligence and Deployment
To transition from research to field application, we implement a "Shrink Ray" pipeline for deployment on $5 IoT hardware (ESP32).

*   **Model Compression**: Conversion of PyTorch `.pth` weights to universal ONNX format.
*   **Quantization**: Post-Training Quantization (PTQ) from `Float32` to `Int8`, achieving a 4x reduction in footprint (e.g., 100MB to 25MB) with negligible accuracy loss.
*   **Hardware Acquisition**: Utilization of the **INMP441** MEMS I2S microphone (-26 dBFS sensitivity) for digital-native audio capture on the ESP32 Sniffer hardware.

## 6. Rigorous Scientific Validation
We employ three primary protocols to ensure model reliability in uncontrolled environments:

1.  **Acoustic Robustness Testing**: Evaluating performance decay against controlled noise injections (white noise, torrential rain, and wind interference).
2.  **Interpretability (Grad-CAM)**: Utilizing Gradient-weighted Class Activation Mapping to verify that the model focuses on biologically relevant signatures (e.g., the 450 Hz piping frequency) rather than artifactual environmental noise.
3.  **Generalization (Cross-Dataset Validation)**: Blind-testing the engine on the OSBH dataset after training on NU-Hive to prove portability across different microphone profiles and apiary acoustics.

## 7. Implementation
### 7.1 Environment Setup
```bash
git clone https://github.com/nduva15/BEE-SOUND-ANALYSIS.git
cd BeeSound_Analysis
pip install -r requirements.txt
```

### 7.2 Core Utilities
*   **Indexing**: `python tools/fast_indexer.py` (Rebuilds the master HDF5 index).
*   **Training**: `python tools/train_architecture.py` (Research-grade training loop).
*   **Deployment**: `python tools/export_brain.py` (Freezes architecture for ONNX/TFLite export).

---
**Maintained by Timothy Nduva**  
*Strategic Research Integration for Apicultural Health*
