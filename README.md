# microCube (Bio-Lattice) - v2

[![Release](https://img.shields.io/github/v/release/msancheza/biolattice?color=green)](https://github.com/msancheza/biolattice/releases/latest)
[![License](https://img.shields.io/github/license/msancheza/biolattice)](https://github.com/msancheza/biolattice/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

## Overview

Bio-Lattice is a **Data Representation Framework** for 4D Medical Imaging. 

*   **What it is:** An extraction pipeline that converts complex, multi-phase breast MRI sequences (DICOM) into standardized 3D tensors.
*   **What it produces:** The **Micro-Cube**, a compact tensor representation that merges spatial anatomy and temporal dynamics.
*   **Why it matters:** It abstracts away the heavy engineering complexity of native 4D medical imaging (registration, metadata, padding), allowing data scientists to plug clean data directly into downstream models.

## Core Concept: The Micro-Cube

Rather than feeding raw slices into a network, Bio-Lattice constructs an explicit `[3, 64, 64, 64]` representation tensor featuring:
1.  **C1 (Anatomy):** Post-contrast structural topology.
2.  **C2 (Heterogeneity):** Local intensity variance.
3.  **C3 (Kinetics):** Contrast wash-in dynamics, computed via sub-voxel rigid registration (FFT) between pre- and post-contrast phases.
4.  **Embedded Physical Metadata:** Key acquisition parameters (`PixelSpacing`, `SliceThickness`) attached to each tensor object.

## What's in this repository

This repository provides everything needed to configure, generate, and validate the representations:
*   **Extraction Engine (`main.py`):** The core DICOM parser, registration, QA gating, and tensor assembly pipeline.
*   **Validation Sandbox (`train.py`):** An embedded 3D-ResNet reference model. Included strictly as a test framework to empirically evaluate whether the extracted Micro-Cubes retain predictive signal relevant to the target task. 
*   **Explainability Interface (`visualizer.py` & `dashboard/app.py`):** A diagnostic UI featuring Grad-CAM 3D, designed to visually map the network's focal points against the extracted clinical channels.

## The Workflow

Bio-Lattice is composed of three interconnected modules that govern data from raw MRI ingestion to clinical verification:

### 1. Data Representation & Quality Assurance
The core engine processes and standardizes raw DICOM sets. Because clinical data is inherently noisy, this stage incorporates a strict **Quality Assurance (QA) Gate**:
*   **Biological Targeting:** Chronologically isolates exact pre-contrast and peak-enhancement sequences to avoid kinetic dilution.
*   **Geometrical Integrity:** Corrects MRI slope/intercept distortions and detects spatial Z-gap discontinuities.
*   **Mathematical Alignment:** Performs sub-voxel FFT registration to correct for respiratory motion between phases.
*   **Audit Segregation:** Cases with irrecoverable motion or missing temporal phases are actively segregated (`REVIEW`) to prevent dataset pollution.

### 2. Validation Sandbox
To evaluate the predictive performance of the extracted Micro-Cubes, the framework consumes the curated dataset and performs supervised learning using a specified ground truth. It dynamically handles class balancing, computes Focal Loss, and enforces Early Stopping, ensuring a robust baseline evaluation

### 3. Clinical Explainability
To provide visual interpretability signals, the inference module generates probabilistic risk assessments, while the companion **Expert Console** computes attention heatmaps. Investigators can visually cross-reference the network's focal points directly against the isolated Anatomy, Heterogeneity, and Kinetic channels.

## Clinical Framing: From Detection to Phenotyping

When testing the framework using the Duke Breast MRI dataset, the task must be framed correctly:

*   **Oncologic Data Reality:** The Duke dataset is a purely oncological cohort. Training a standard algorithm to "detect" cancer in a database where all patients are confirmed positive is not mathematically viable.
*   **Strategic Pivot:** Bio-Lattice acts as a **Virtual Risk Phenotype** representation layer. Rather than detecting cancer, the validation sandbox estimates a tumor's biological aggressiveness, aligning its predictions with the **Molecular Subtype** reported by pathology.
*   **Architecture Scalability:** When applied to distinct medical datasets containing healthy controls, this same representation framework can support traditional diagnosis or triage tasks without modifying its extraction logic.

## Training Details & Ground Truth

The validation sandbox uses the **`Mol Subtype`** column from the clinical file as its training target.

| Code label | Rule in `helper.BioLatticeDataset` | Clinical Meaning |
|------------|-----------------------------------|------------------|
| **0** | `Mol Subtype = 0` | **Lower Risk** (Luminal A) |
| **1** | `Mol Subtype > 0` | **Higher Risk** (Luminal B, HER2+, Triple Negative) |

*   **Optimization:** Focal Loss emphasizes hard examples. The image embeddings are concatenated with a small MLP processing the embedded physical metadata (`PixelSpacing` + `SliceThickness`).
*   **Performance Tracking:** Telemetry is written securely to `dashboard/training_logs/` providing epoch-by-epoch loss tracking without third-party dependencies.

## Setup & Usage

### 1. Dataset Access (Duke Breast MRI)
The foundational data for this project originates from the public **Duke Breast Cancer MRI dataset** (via TCIA). Because raw DICOM MRI sequences require hundreds of gigabytes of storage, they are purposely gitignored. 

To bypass the heavy DICOM extraction phase, you can download the pre-computed Micro-Cubes directly from Hugging Face:
🔗 **[HuggingFace: msancheza/microCube-Duke-Breast-MRI](https://huggingface.co/datasets/msancheza/microCube-Duke-Breast-MRI)**

Place the downloaded `*_lattice.pt` files into your local `datasets/micro_cubes/` directory to run inference or training immediately.

### 2. Requirements and Installation
* Python 3.10+
* *(If extracting from scratch)*: Duke Cohort data structure (`datasets/raw_data/`, `datasets/Annotation_Boxes.xlsx`, `datasets/Clinical_and_Other_Features.xlsx`).
```bash
git clone https://github.com/msancheza/biolattice.git
cd biolattice
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Execution Operations
*   **Configure:** Adjust parameters, paths, and hardware targets centrally in `config.py`.
*   **Extract:** Run `python main.py` to walk the DICOM tree, enforce the Quality Gate, and generate `*_lattice.pt` capsules into `datasets/micro_cubes/`.
*   **Validate:** Run `python train.py` to benchmark the capsules by training the ResNet sandbox layer.
*   **Explore:** Launch `streamlit run dashboard/app.py` to use the interactive orchestrator, analyze metrics, and invoke the Grad-CAM visualizer.

## Green AI & Computational Efficiency

Instead of training massive 3D Convolutional Networks directly on gigabytes of DICOMs, Bio-Lattice mathematically condenses imaging data from gigabyte-scale DICOM series to megabyte-scale tensors prior to deep learning. This allows the sandbox to train natively on consumer-grade hardware (e.g., Apple Silicon) in minutes rather than days on cloud GPUs, drastically reducing operational carbon footprints.

<div align="center">
🔬 Research prototype only — not for clinical use
</div>
