# microCube (Bio-Lattice)

Converts raw breast MRI volumes (DICOM) into highly compact **32×32×32 4D micro-cubes** with **3 independent channels** (structural foundation, 3D texture variance/GLCM, and pre/post contrast kinetics). Currently, this codebase trains a custom **3D-ResNet neural network** over these tensors for a specialized clinical binary classification task: **Benign vs. Malignant**. 

The micro-cube itself is a powerful **input representation**: with different clinical labels and a modified classification head (e.g., multi-class molecular subtypes), it could be adapted to other diagnostic tasks, subject to cohort size and data availability.

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
    %% Estilos Globales
    classDef raw fill:#eef2f5,stroke:#93a1a1,stroke-width:2px,color:#2c3e50;
    classDef process fill:#fef9e7,stroke:#d4ac0d,stroke-width:2px,color:#7d6608;
    classDef tensor fill:#e8f8f5,stroke:#28b463,stroke-width:2px,color:#145a32,font-weight:bold;
    classDef net fill:#fdf2e9,stroke:#e74c3c,stroke-width:2px,color:#78281f;

    %% 1. DICOM crudos
    subgraph Fase 1: Duke MRI Dataset
        PRE["Fase Pre-Contraste <br/> V_pre"]:::raw
        POST["Fase Post-Contraste <br/> V_post"]:::raw
    end

    %% 2. Extracción Geométrica
    subgraph Fase 2: Localización y Registro
        ALIGN{"Co-Registro Espacial <br/> Mitigación de Artefactos"}:::process
        ROI["Extracción ROI Bounding Box <br/> + Context Padding Halo 20%"]:::process
    end

    PRE --> ALIGN
    POST --> ALIGN
    ALIGN --> ROI

    %% 3. La Red de Bio-Lattice
    subgraph Fase 3: Forjado Multimodal 4D - El Tejedor
        C1["Canal 1: Estructura <br/> Adaptive Max Pooling"]:::process
        C2["Canal 2: Varianza Radiómica <br/> E X² - E X ²"]:::process
        C3["Canal 3: Cinética Temporal <br/> Sustracción V_post - V_pre"]:::process
    end

    ROI --> C1
    ROI --> C2
    ROI --> C3

    %% 4. Salida Final
    CUB{"(Bio-Lattice Tensor) <br/> 3 Canales x 32³ Píxeles"}:::tensor
    C1 --> CUB
    C2 --> CUB
    C3 --> CUB

    %% 5. Inferencia
    RES(["Arquitectura 3D-ResNet"]):::net
    LOSS("Calibración: Biopsia Real <br/> BCEWithLogitsLoss")

    CUB ==> RES
    RES -.-> LOSS
```

2. **`python train.py`** — Trains the `RedMicroCubo3Ch` residual classifier natively and saves the optimal model weights to `datasets/modelo/biolattice_3dresnet_binary.pth`.
3. **`python predict.py`** — Performs Virtual Biopsy inference for a specific `Patient ID`.
4. **`streamlit run dashboard/app.py`** — Launches the interactive UI orchestrator to handle the full pipeline and dataset evaluations visually.

## Additional Details

In **`docs/`** you'll find the technical pipeline documentation and business validation notes. **`visualizer.py`** provides utilities to inspect the geometry of a generated cube. **`backup/`** keeps legacy script versions.

## Medical Disclaimer

This is strictly a **Research Prototype**, not a certified medical device. Do not use for final clinical decisions or standalone patient diagnosis.
