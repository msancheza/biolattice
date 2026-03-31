# microCube (Bio-Lattice)

Converts raw breast MRI volumes (DICOM) into highly compact **32×32×32 4D micro-cubes** with **3 independent channels** (structural foundation, 3D texture variance/GLCM, and pre/post contrast kinetics). Currently, this codebase trains a custom **3D-ResNet neural network** over these tensors for a specialized clinical binary classification task: **Benign vs. Malignant**. 

The micro-cube itself is a powerful **input representation**: with different clinical labels and a modified classification head (e.g., multi-class molecular subtypes), it could be adapted to other diagnostic tasks, subject to cohort size and data availability.

## 🌱 Green AI & Computational Efficiency

Instead of training massive, energy-hungry 3D Convolutional Networks directly on gigabyte-scale DICOMs, Bio-Lattice mathematically condenses clinical data into microscopic 4D tensors *prior* to deep learning. This allows the core 3D-ResNet to train natively on consumer-grade hardware (e.g., Apple Silicon) in minutes rather than days on cloud GPUs. This architecture drastically reduces the operational carbon footprint and cloud computing costs, democratizing high-tier medical research without sacrificing diagnostic sensitivity.

## Requirements

- Python 3.10+ (The project locally uses Python 3.13)
- Duke Cohort type data: `datasets/raw_data/<PatientID>/...`, `datasets/Annotation_Boxes.xlsx`, `datasets/Clinical_and_Other_Features.xlsx`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PyTorch: If you need a specific hardware variant (CPU/CUDA), follow the [official installation guide](https://pytorch.org/get-started/locally/).

## Usage (Core Pipeline)

1. **`python main.py`** — Parses DICOMs, crops the target Region of Interest (with padding), and serializes `.pt` tensors into `datasets/micro_cubos/`.

```mermaid
graph TD
    %% Global Styles
    classDef raw fill:#eef2f5,stroke:#93a1a1,stroke-width:2px,color:#2c3e50;
    classDef process fill:#fef9e7,stroke:#d4ac0d,stroke-width:2px,color:#7d6608;
    classDef tensor fill:#e8f8f5,stroke:#28b463,stroke-width:2px,color:#145a32,font-weight:bold;
    classDef net fill:#fdf2e9,stroke:#e74c3c,stroke-width:2px,color:#78281f;

    %% 1. Raw DICOM Sequences
    subgraph Phase 1: Duke MRI Dataset
        PRE["Pre-Contrast Phase <br/> V_pre"]:::raw
        POST["Post-Contrast Phase <br/> V_post"]:::raw
    end

    %% 2. Geometric Extraction
    subgraph Phase 2: Localization & Registration
        ALIGN{"Spatial Co-Registration <br/> Artifact Mitigation"}:::process
        ROI["ROI Bounding Box Extraction <br/> + 20% Context Padding Halo"]:::process
    end

    PRE --> ALIGN
    POST --> ALIGN
    ALIGN --> ROI

    %% 3. The Tensor Weaver
    subgraph Phase 3: 4D Multi-Modal Forging - The Weaver
        C1["Channel 1: Structure <br/> Adaptive Max Pooling"]:::process
        C2["Channel 2: Radiomic Variance <br/> E X² - E X ²"]:::process
        C3["Channel 3: Temporal Kinetics <br/> Subtraction V_post - V_pre"]:::process
    end

    ROI --> C1
    ROI --> C2
    ROI --> C3

    %% 4. Final Output
    CUB{"(Bio-Lattice Tensor) <br/> 3 Channels x 32³ Pixels"}:::tensor
    C1 --> CUB
    C2 --> CUB
    C3 --> CUB

    %% 5. Downstream Inference
    RES(["3D-ResNet Architecture"]):::net
    LOSS("Ground Truth Calibration <br/> Biopsy via BCEWithLogitsLoss")

    CUB ==> RES
    RES -.-> LOSS
```

2. **`python train.py`** — Trains the `RedMicroCubo3Ch` residual classifier natively and saves the optimal model weights to `datasets/modelo/biolattice_3dresnet_binary.pth`.
3. **`python predict.py`** — Performs Virtual Biopsy inference for a specific `Patient ID`.
4. **`streamlit run dashboard/app.py`** — Launches the interactive UI orchestrator to handle the full pipeline and dataset evaluations visually.


## Medical Disclaimer

This is strictly a **Research Prototype**, not a certified medical device. Do not use for final clinical decisions or standalone patient diagnosis.
