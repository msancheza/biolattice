# Contributing to Bio-Lattice (microCube) 

Thank you for your interest in contributing! Bio-Lattice is an open-source research framework designed to bridge the gap between complex 4D medical imaging and deep learning. 

By contributing, you help make medical AI more accessible, transparent, and efficient.

---

## How You Can Contribute

### 1. Bug Reports & Improvements
If you find a bug (e.g., DICOM parsing errors, registration artifacts), please open an **Issue** with:
*   A clear description of the problem.
*   Steps to reproduce.
*   The version of the framework (v2.x).

### 2. Feature Requests
We welcome ideas for:
*   **New Channels:** Better texture extraction or kinetics logic.
*   **Multimodal Support:** Extending the Micro-Cube to DWI, T2, or PET-CT.
*   **Performance:** Speeding up the FFT registration engine.

### 3. Pull Requests (PRs)
Ready to write code? Great! Please follow these steps:
1.  **Fork** the repository and create your branch (`feature/amazing-feature`).
2.  Ensure your code follows **PEP8** standards.
3.  Add **Docstrings** and comments to explain complex mathematical logic.
4.  Run existing extraction tests to ensure no regression in signal integrity.
5.  Open a **PR** with a detailed summary of your changes.

---

## Development Setup

1.  Clone your fork: `git clone https://github.com/msancheza/biolattice.git`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Set up the folder structure: ensure you have a `datasets/raw_data/` folder for testing.

---

## Research Integrity & Values

Bio-Lattice is built on **Green AI** principles. We prioritize:
*   **Efficiency:** Small, dense tensors over massive, redundant volumes.
*   **Explainability:** Every channel must have a clear clinical meaning.
*   **Transparency:** No "black box" pre-processing. All signal transformations must be auditable.

---

## Code of Conduct

Please be respectful and professional. We are a community of researchers and developers working towards a common goal: improving healthcare through technology.

---

> [!IMPORTANT]
> **Data Privacy:** Never upload real patient DICOM data or identifying information to GitHub. Use anonymized datasets like the Duke Breast MRI dataset for testing and benchmarking.
