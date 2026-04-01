# microCube (Bio-Lattice)

Converts raw breast MRI volumes (DICOM) into highly compact **32×32×32** tensors with **3 channels**: post-contrast **structure** (adaptive max pool), **local heterogeneity** (pooled variance **E[X²]−E[X]²** per micro-cell—related to texture but **without** explicit GLCM or LBP), and **kinetics** (pooled post − pre). This codebase trains a custom **3D-ResNet** on those tensors for a **binary target** described in clinical shorthand as *benign vs. malignant*—but **operationally defined from the Duke spreadsheet** (see [Training labels](#training-labels-ground-truth) below).

The micro-cube itself is a powerful **input representation**: with different clinical labels and a modified classification head (e.g., multi-class molecular subtypes), it could be adapted to other diagnostic tasks, subject to cohort size and data availability.

### Training labels (ground truth)

Training and evaluation use **`datasets/Clinical_and_Other_Features.xlsx`** (loaded with `header=1`, as in the Duke TCIA companion table). **Only rows with a non-null `Mol Subtype` value are kept**; patients without that field never enter the training set.

| Code label | Rule in `train.py` | Meaning |
|------------|-------------------|---------|
| **0** | `Mol Subtype ≤ 0` | “Negative” class in experiments |
| **1** | `Mol Subtype > 0` | “Positive” class in experiments |

This is a **pragmatic proxy**: it maps a numeric **molecular subtype** column to a binary target. It is **not** wired to a free-text pathology or imaging report inside this repo. Metrics (AUC, accuracy, etc.) therefore measure separability under **this rule**, not an abstract gold standard for every possible definition of malignancy. For other institutions or papers, swap in labels from your own approved reference (e.g., biopsy-confirmed BIRADS, pT stage) and adjust `BioLatticeDataset` accordingly.

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

## Dataset & Reproducibility (Duke Breast Cancer MRI)

For testing was used the public **Duke Breast Cancer MRI dataset** from The Cancer Imaging Archive (TCIA). 

Since raw DICOM MRI sequences and the generated 4D tensors weigh hundreds of gigabytes, **they are not included in this code repository** to ensure lightning-fast cloning. 

* **To test the pipeline out-of-the-box:** You can use the pre-compiled tensors provided in the `datasets/examples_microcubos/` folder. Just copy those few `.pt` files into `datasets/micro_cubos/` to instantly run the *Predict* (`predict.py`) or test the Streamlit dashboard on patients `Breast_MRI_001` through `007` without needing to download massive medical archives.
* **To reproduce the full research:** You must download the native Duke cohort from TCIA, extract the DICOMs into `datasets/raw_data/<PatientID>/`, and run the Data Extraction Module (`main.py`) to generate your own tensors.

## Usage (Core Pipeline)

1. **`python main.py`** — Builds `datasets/micro_cubos/<PatientID>_lattice.pt` (see below for **exact** behavior and limits).

### What `main.py` actually does

This pipeline is optimized for a **Duke-style TCIA layout** and raw `pydicom` loading. It is **not** a full clinical preprocessing stack.

| Step | Reality |
|------|--------|
| **Series choice** | Walks each patient folder, reads one DICOM per subfolder for `SeriesDescription`, then picks **pre** vs **post** with **substring rules** tuned to Duke naming (`pre`, `dyn` without “phase”, vs `1st` / `ph1` / etc.). If several folders match, the **last** match in the walk **wins**—there is no tie-break or UI. Other sites will usually need new rules or manual series mapping. |
| **3D stack** | Slices are sorted by **`ImagePositionPatient[2]`** and stacked as **`pixel_array`** (`float32`). **`RescaleSlope` / `RescaleIntercept`** are **not** applied here; downstream training applies **per-cube Z-score**, not absolute intensity calibration. |
| **ROI** | Box comes from **`Annotation_Boxes.xlsx`**: `Start Slice/Row/Column` are treated as **1-based** and shifted by **−1** for NumPy; **`End Slice/Row/Column`** feed Python slice ends as in the current code (verify against your TCIA column definitions if you change cohorts). The same integer box is applied to **both** phases **before** any resize. |
| **Pre vs post geometry** | There is **no rigid or deformable registration**. After crop, if pre and post tensors differ in shape, **`weave_4d_micro_cube`** only **trilinearly resizes the pre ROI** to match the post ROI volume (see `main.py`). That aligns **grid size**, not guaranteed **anatomy**. Channel 3 (**post − pre**) assumes phases are already comparable within the ROI (as in many research datasets with fixed protocols). |
| **Output** | A single **`[3, 32, 32, 32]`** tensor per patient: pooled post-contrast structure, pooled local variance on post, and pooled **post − pre** kinetics. |

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

    %% 2. Pairing + ROI (no rigid registration in code)
    subgraph Phase 2: Phase pairing and ROI
        SEL["PRE/POST folders <br/> SeriesDescription heuristics (Duke)"]:::process
        ROI["Same 3D box + ~20% halo <br/> from Annotation_Boxes.xlsx"]:::process
    end

    PRE --> SEL
    POST --> SEL
    SEL --> ROI

    %% 3. The Tensor Weaver
    subgraph Phase 3: Compress to 32³ (weave)
        C1["Channel 1: Structure <br/> Adaptive Max Pooling"]:::process
        C2["Channel 2: Local heterogeneity <br/> pooled var E[X²]−E[X]²"]:::process
        C3["Channel 3: Kinetics <br/> pooled post − pre (pre resized if shape mismatch)"]:::process
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
    LOSS("Supervision: Duke Mol Subtype <br/> binary rule + BCEWithLogitsLoss")

    CUB ==> RES
    RES -.-> LOSS
```

2. **`python train.py`** — Trains the `BioLattice3DResNet` residual classifier natively and saves the optimal model weights to `datasets/modelo/biolattice_3dresnet_binary.pth`.
3. **`python predict.py`** — Performs Virtual Biopsy inference for a specific `Patient ID`.
4. **`streamlit run dashboard/app.py`** — Launches the interactive UI orchestrator to handle the full pipeline and dataset evaluations visually.


## Medical Disclaimer

This is strictly a **Research Prototype**, not a certified medical device. Do not use for final clinical decisions or standalone patient diagnosis.
