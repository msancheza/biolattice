import streamlit as st
import subprocess
import os
import sys

# Page Configuration
st.set_page_config(page_title="Bio-Lattice 4D", layout="wide", page_icon="🧬")

# Custom CSS for minimalist, smaller typography (Dark Mode compatible)
st.markdown("""
    <style>
    /* Base font settings */
    html, body, [class*="st-"] {
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        font-size: 13px !important;
        font-weight: 300;
    }
    
    /* Headers */
    h1 { font-size: 22px !important; font-weight: 500 !important; letter-spacing: -0.5px; padding-bottom: 5px;}
    h2 { font-size: 16px !important; font-weight: 500 !important; }
    h3 { font-size: 14px !important; font-weight: 400 !important; }
    
    /* Buttons */
    .stButton>button { 
        border-radius: 4px; 
        font-weight: 400; 
        font-size: 13px; 
    }
    
    /* Minimalist tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 0px;
        font-size: 14px;
        font-weight: 400;
        opacity: 0.8;
    }
    .stTabs [aria-selected="true"] {
        border-bottom-color: var(--primary-color) !important;
        opacity: 1;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Bio-Lattice 4D Dashboard")
st.markdown("Minimalist orchestration engine for the 3D-ResNet tumor classification pipeline.")
st.divider()

# Project Root Directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_script(script_name):
    """Executes a target python script from the root project directory."""
    try:
        result = subprocess.run(
            ["python", script_name], 
            cwd=BASE_DIR, 
            capture_output=True, 
            text=True
        )
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)

# --- STEP-BY-STEP TABS ---
tabs = st.tabs([
    "ℹ️ About Project", 
    "1. Data Extraction •", 
    "2. Model Training •", 
    "3. Validation Metrics •", 
    "4. Patient Inference"
])
tab_about, tab1, tab2, tab3, tab4 = tabs

# TAB 0: ABOUT / README
with tab_about:
    st.markdown("## 🧬 Bio-Lattice 4D: Project Overview")
    st.markdown("""
    **Bio-Lattice 4D** (*microCube*) converts raw breast MRI volumes (DICOM) into highly compact **32×32×32 4D micro-cubes**. 
    These volumetric tensors capture the tumor's foundational structure, **local heterogeneity** (pooled **E[X²]−E[X]²** per micro-cell—texture-like signal without explicit GLCM/LBP), and pre/post contrast kinetics across **3 independent spatial channels**.
    
    The orchestrator trains a custom **3D-ResNet Deep Learning architecture** over these tensors to perform a specialized clinical binary classification task: **Benign vs. Malignant**.
    
    ### ⚙️ Pipeline Lifecycle:
    1. **Data Extraction:** Parses DICOM cohort sequences, algorithmically crops the tumor Region of Interest (ROI), and serializes dense `.pt` tensors.
    2. **Model Training:** Dynamically trains the `BioLattice3DResNet` residual classifier efficiently leveraging Apple Silicon (MPS) hardware acceleration, Z-Score distribution modeling, and aggressive False Negative penalization thresholds.
    3. **Clinical Validation:** Evaluates inference strictness against an isolated 20% dataset to retrieve global Accuracy, ROC AUC, Sensitivity, and Specificity metrics.
    4. **Virtual Biopsy (Inference):** Predicts malignancy risk probability pixel-by-pixel on single un-seen patient tensors natively.
    
    > **⚠️ Medical Disclaimer:**
    > This orchestrator and its underlying diagnostic algorithms are strictly a **Research Prototype**. It is not a certified medical device and must never be utilized for final clinical decisions or standalone patient diagnosis.
    """)

# TAB 1: DATA EXTRACTION
with tab1:
    st.markdown("### Prepare 4D Micro-Cubes")
    st.markdown("Processes raw MRI scans into volumetric tensors: ROI crop, co-register pre/post to a common grid, then three adaptive-pooling channels (structure, pooled variance, wash-in).")
    
    st.write("") # Spacer
    if st.button("Run Data Extraction (main.py)", use_container_width=True):
        with st.spinner("Extracting features and generating tensors..."):
            stdout, stderr = run_script("main.py")
        st.success("Data Extraction completed successfully.")
        with st.expander("Show Console Logs"):
            st.code(stdout + "\n" + stderr)

# TAB 2: MODEL TRAINING
with tab2:
    st.markdown("### Train 3D-ResNet Model")
    st.markdown("Executes the deep learning neural network cycle. Supports Apple Silicon MPS & NVIDIA CUDA.")
    
    try:
        if BASE_DIR not in sys.path:
            sys.path.append(BASE_DIR)
        from train import EPOCHS, BATCH_SIZE
        st.info(f"⚙️ Configured for **{EPOCHS} Epochs** (Batch Size: {BATCH_SIZE}).")
    except Exception:
        pass
        
    st.write("") # Spacer
    if st.button("Start Diagnostics Training (train.py)", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_expander = st.empty()
        
        with st.spinner("Initializing Neural Engine..."):
            try:
                import re
                
                # Exect python with -u (unbuffered) to force logs in real time
                process = subprocess.Popen(
                    ["python", "-u", "train.py"], 
                    cwd=BASE_DIR, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1
                )
                
                full_log = ""
                for line in process.stdout:
                    full_log += line
                    
                    # Lógica de barra de progreso para "Epoch [X/Y]"
                    match = re.search(r"Epoch \[(\d+)/(\d+)\]", line)
                    if match:
                        curr_epoch = int(match.group(1))
                        total_ep = int(match.group(2))
                        progress_bar.progress(curr_epoch / total_ep)
                        status_text.markdown(f"**Training Live:** Epoch `{curr_epoch}` of `{total_ep}`")
                
                process.wait()
                
                if process.returncode == 0:
                    status_text.markdown("✅ **Training complete.**")
                    st.success("Model correctly trained and weights updated locally.")
                else:
                    st.error("Training encountered an error or was aborted.")
                    
                with st.expander("Show Full Training Logs", expanded=False):
                    st.code(full_log)
                    
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
            finally:
                # Prevenir procesos huérfanos forzando un SIGKILL si el usuario detiene el dashboard
                if 'process' in locals() and process.poll() is None:
                    try:
                        process.kill()
                        process.wait(timeout=2)
                    except Exception:
                        pass

# TAB 3: VALIDATION
with tab3:
    st.markdown("### Clinical Validation")
    st.markdown("Calculates absolute global precision across the strictly separated 20% validation dataset.")
    
    st.write("") # Spacer
    if st.button("Evaluate Global Metrics", type="primary", use_container_width=True):
        with st.spinner("Performing widespread inference on unseen medical data..."):
            try:
                if BASE_DIR not in sys.path:
                    sys.path.append(BASE_DIR)
                
                # Import dynamically to ensure the latest weights are evaluated 
                from predict import evaluate_dataset
                
                res = evaluate_dataset()
                
                if "error" in res:
                    st.error(res["error"])
                else:
                    st.markdown(f"**Results based on {res['total']} isolated validation patients:**")
                    
                    # Native minimal metrics
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", f"{res['accuracy']*100:.1f}%")
                    c2.metric("ROC AUC", f"{res['auc']:.3f}")
                    c3.metric("Sensitivity", f"{res['sensibilidad']*100:.1f}%")
                    c4.metric("Specificity", f"{res['especificidad']*100:.1f}%")
                    
                    st.divider()
                    st.markdown("**Confusion Matrix**")
                    col_tn, col_fp, col_fn, col_tp = st.columns(4)
                    col_tn.metric("True Negatives", res['matriz']['tn'])
                    col_fp.metric("False Positives", res['matriz']['fp'])
                    col_fn.metric("False Negatives", res['matriz']['fn'])
                    col_tp.metric("True Positives", res['matriz']['tp'])
                    
            except Exception as e:
                st.error(f"Failed to evaluate dataset: {str(e)}")

# TAB 4: INDIVIDUAL PATIENT INFERENCE
with tab4:
    st.markdown("### Patient Inference")
    st.markdown("Run the Bio-Lattice Virtual Biopsy on an individual patient's 4D tensor.")
    
    st.write("") # Spacer
    patient_id = st.text_input("Enter Patient ID:", placeholder="Breast_MRI_002")
    
    if st.button("Run Prediction (predict.py)", type="primary", use_container_width=True):
        if not patient_id.strip():
            st.warning("Please enter a valid Patient ID first.")
        else:
            with st.spinner(f"Analyzing 4D Biopsy for {patient_id}..."):
                try:
                    if BASE_DIR not in sys.path:
                        sys.path.append(BASE_DIR)
                        
                    from predict import predict_patient
                    res = predict_patient(patient_id.strip())
                    
                    if not res:
                        st.error("Unknown error executing prediction script.")
                    elif "error" in res:
                        st.error(res["error"])
                    else:
                        st.markdown(f"**Results for Patient: `{patient_id}`**")
                        
                        colA, colB = st.columns(2)
                        
                        # Big visual indicator
                        if res["positivo"]:
                            colA.error("🚨 POSITIVE (HIGH MALIGNANCY RISK)")
                        else:
                            colA.success("✅ NEGATIVE (LIKELY BENIGN TISSUE)")
                            
                        # Metric gauge
                        colB.metric("Malignancy Probability", f"{res['riesgo']:.2f}%")
                        st.caption(f"Configured Positive Threshold: ≥ {res['umbral']:.0f}%")
                except Exception as e:
                    st.error(f"Failed to execute patient prediction: {str(e)}")
