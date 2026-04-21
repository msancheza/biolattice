# Bio-Lattice (microCube) - v2

[![Version](https://img.shields.io/badge/version-2.0--alpha-orange)](https://github.com/msancheza/biolattice)
[![License](https://img.shields.io/github/license/msancheza/biolattice)](https://github.com/msancheza/biolattice/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

## Bio-Lattice: A proposed representation for 4D-MRI

Bio-Lattice is a research prototype for transforming 4D DCE-MRI sequences into compact tensor representations (**Micro-Cubes**), designed to provide a consistent input structure for downstream models. 

Each Micro-Cube is a multi-channel tensor with shape **(4, 64, 64, 64)**, capturing spatial ($64^3$), morphological, and hemodynamic patterns within a unified structure.

The system implements a four-channel extraction engine representing different radiological markers:
- **Channel 1 (Anatomy):** Structural representation through post-contrast intensity averages.
- **Channel 2 (Variability):** Local variance map to quantify tissue heterogeneity.
- **Channel 3 (Kinetics):** Voxel-wise map of signal enhancement, quantifying the magnitude of the 'brightening' effect between phases (Log-Relative Change).
- **Channel 4 (Vascular Peaks):** Isolation of peak enhancement signals against the structural average.

### Workflow application
The tool enables the use of 4D data in standard deep learning models by reducing data volume from gigabytes to megabytes through structured tensor representation, while preserving spatiotemporal structure.

The project uses this method to explore relationships between imaging features and molecular subtypes in oncological datasets.

## Repository Contents

This framework provides tools for representation generation and validated benchmarking:
*   **Extraction Engine (`main.py`):** The core DICOM parser, semantic classifier, registration suite, QA gating, and tensor assembly pipeline.
*   **Validation Sandbox (`train.py`):** A 3D-ResNet reference model included as a systematic test framework to evaluate whether the extracted Micro-Cubes retain predictive signal relative to target labels.
*   **Evaluation Interface (`dashboard/app.py`):** A research UI featuring Grad-CAM 3D, designed to map network attention against the isolated Anatomy, Heterogeneity, and Kinetic channels.

## System Architecture

The workflow is divided into three modules:

### 1. Extraction and Quality Control
The pipeline standardizes raw DICOM data and implements a systematic registration process:
*   **Semantic Classification:** Employs a metadata-aware engine (`SeriesClassifier`) with vendor-specific rules (GE, Siemens, Philips) to intelligently identify PRE and POST sequences.
*   **Registration:** Uses FFT-based phase correlation to align volumes, accounting for patient movement between series.
*   **Padding Strategy:** Applies conditional padding to maintain a consistent region-of-interest (ROI) shape.
*   **Audit Logging:** Generates a structured audit trail (JSONL) to track registration quality and data consistency.

### 2. Benchmark Environment
Includes a testing framework to evaluate if the extracted tensors retain predictive signals. This module implements training procedures such as Focal Loss and class balancing to handle imbalanced medical datasets.

### 3. Analysis Interface
Provides a tool to visualize model attention maps (Grad-CAM 3D) alongside individual Micro-Cube channels, facilitating the review of which anatomical or kinetic features influence the outputs.

## Research Context: Molecular Subtype Estimation

The current implementation utilizes the Duke Breast MRI dataset with the following focus:

*   **Task Definition:** The validation objective is to evaluate the correlation between extracted imaging features and reported molecular subtypes.
*   **Methodological Role:** The Micro-Cube serves as a standardized input for downstream classification tasks, comparing image-derived patterns against known biological markers.

## Preliminary Results

Preliminary benchmarking using a 3D-ResNet architecture demonstrates that Micro-Cube representations retain significant predictive signal, achieving an **ROC-AUC of 0.7447** on the Duke Breast MRI cohort. The model exhibits a higher sensitivity (0.81) relative to its specificity (0.50), indicating a classification bias towards recall in this experimental configuration.

## Training Configuration & Benchmarking

The validation sandbox utilizes the **`Mol Subtype`** targets for benchmarking purposes.

| Target Category | Code | Clinical Context (Reference) |
|-----------------|------|-----------------------------|
| **Lower Risk** | `0` | Luminal A |
| **Higher Risk** | `1` | Luminal B, HER2+, Triple Negative |

*   **Optimization:** Focal Loss is used to emphasize difficult-to-classify samples. Telemetry is recorded locally in `dashboard/training_logs/` for offline performance analysis.

## Setup & Execution

### 1. Requirements
* Python 3.10+
* Dataset Structure: Duke Cohort format (DICOM folders + Annotation/Clinical Excel files).

### 2. Installation
```bash
git clone https://github.com/msancheza/biolattice.git
cd biolattice
pip install -r requirements.txt
```

### 3. Operations
*   **Extract:** `python main.py` to generate `.pt` capsules from raw DICOMs.
*   **Validate:** `python train.py` to run the ResNet benchmark.
*   **Analyze:** `streamlit run dashboard/app.py` to explore metrics and visual heatmaps.

## Computational Efficiency

By pre-processing DICOM series into compact tensors, the workflow reduces data dimensionality before training. This reduction in memory and storage requirements allows for more efficient iterations with 3D architectures compared to training on raw volumetric sequences.

---
<div align="center">
🔬 <i>Research prototype — not for clinical or diagnostic use.</i>
</div>
