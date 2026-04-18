import streamlit as st
import subprocess
import os
import sys
import numpy as np

# Page Configuration
st.set_page_config(page_title="Bio-Lattice 4D", layout="wide", page_icon="🧬")

# Custom CSS for minimalist typography
st.markdown("""
    <style>
    html, body, [class*="st-"] {
        font-family: 'Inter', 'Helvetica Neue', sans-serif;
        font-size: 13px !important;
        font-weight: 300;
    }
    h1 { font-size: 22px !important; font-weight: 500 !important; letter-spacing: -0.5px; padding-bottom: 5px;}
    h2 { font-size: 16px !important; font-weight: 500 !important; }
    h3 { font-size: 14px !important; font-weight: 400 !important; }
    .stButton>button { border-radius: 4px; font-weight: 400; font-size: 13px; }
    </style>
""", unsafe_allow_html=True)

st.title("Bio-Lattice 4D Dashboard")
st.markdown("Minimalist orchestration engine for the 3D-ResNet tumor classification pipeline.")
st.divider()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_script(script_name):
    try:
        result = subprocess.run(["python", script_name], cwd=BASE_DIR, capture_output=True, text=True)
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)

tabs = st.tabs([
    "ℹ️ About Project",
    "1. Data Extraction",
    "2. Model Training",
    "3. Validation Metrics",
    "4. Patient Inference",
    "5. Expert Visualizer 🔬",
    "6. Config Guide",
])
tab_about, tab1, tab2, tab3, tab4, tab5, tab6 = tabs

with tab_about:
    st.markdown("## 🧬 Bio-Lattice 4D: Project Overview")
    st.markdown("""
    **Bio-Lattice 4D** (*microCube*) converts raw breast MRI volumes (DICOM) into highly compact **64×64×64 4D micro-cubes**. 
    These volumetric tensors capture the tumor's foundational structure, **local heterogeneity**, and pre/post contrast kinetics across **3 independent spatial channels**.
    
    ### ⚙️ Pipeline Lifecycle:
    1. **Data Extraction:** Duke-oriented series selection, isotropic resampling, and ROI engineering. Saves `[3, 64, 64, 64]` tensors with audit trails.
    2. **Model Training:** Trains a multi-modal **3D-ResNet** using image embeddings + physical metadata (spacing/thickness).
    3. **Clinical Validation:** Evaluates performance against a strictly separated 20% dataset (AUC, Youden J).
    4. **Inference:** Diagnostic risk report for individual patients.
    5. **Expert Visualization:** Explainable AI (Grad-CAM) + multi-channel decomposition.
    
    > **⚠️ Medical Disclaimer:**
    > This orchestrator and its underlying diagnostic algorithms are strictly a **Research Prototype**. It is not a certified medical device and must never be utilized for final clinical decisions.
    """)

with tab1:
    st.markdown("### Prepare 4D Micro-Cubes")
    if st.button("Run Data Extraction (main.py)", use_container_width=True):
        with st.spinner("Extracting features..."):
            stdout, stderr = run_script("main.py")
            st.success("Completed.")
            with st.expander("Logs"): st.code(stdout + stderr)

with tab2:
    st.markdown("### Train 3D-ResNet Model")
    if st.button("Start Training (train.py)", use_container_width=True):
        with st.spinner("Training..."):
            stdout, stderr = run_script("train.py")
            st.success("Trained.")
            with st.expander("Logs"): st.code(stdout + stderr)

with tab3:
    st.markdown("### Clinical Validation")
    if st.button("Evaluate Global Metrics", use_container_width=True):
        with st.spinner("Evaluating..."):
            try:
                if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
                import predict, importlib
                importlib.reload(predict)
                res = predict.evaluate_dataset()
                st.json(res)
            except Exception as e:
                st.error(str(e))

with tab4:
    st.markdown("### Patient Inference")
    p_id_inf = st.text_input("Patient ID for Risk Report:", placeholder="Breast_MRI_002", key="inf_pid")
    if st.button("Generate Risk Report", type="primary", use_container_width=True):
        try:
            if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
            import predict, importlib
            importlib.reload(predict)
            res = predict.predict_patient(p_id_inf.strip())
            if "error" in res: st.error(res["error"])
            else:
                st.markdown(f"**Results for Patient: `{p_id_inf}`**")
                cA, cB = st.columns(2)
                if res["high_risk"]: cA.error("🚨 POSITIVE (HIGH MALIGNANCY RISK)")
                else: cA.success("✅ NEGATIVE (LOWER RISK PHENOTYPE)")
                cB.metric("Risk Index", f"{res['risk_percent']:.2f}%")
                
                st.caption(f"Configured Positive Threshold: ≥ {res['threshold_percent']:.0f}%")
        except Exception as e:
            st.error(str(e))

with tab5:
    st.markdown("### 🔬 Expert Visualizer (4-Channel Analysis)")
    p_id_viz = st.text_input("Patient ID for Detailed Analysis:", placeholder="Breast_MRI_400", key="viz_pid")
    if st.button("Launch Expert Analysis", type="primary", use_container_width=True):
        try:
            if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
            import visualizer, importlib
            importlib.reload(visualizer)
            fig = visualizer.visualize_expert_analysis(p_id_viz.strip())
            if fig:
                _, col_viz, _ = st.columns([1, 1, 1])
                with col_viz:
                    st.pyplot(fig)
            else: st.error("Files not found.")
        except Exception as e:
            st.error(str(e))

with tab6:
    st.markdown("### Config Guide")
    try:
        if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
        import config as cfg
        st.markdown(f"**Inference Threshold:** `{cfg.MALIGNANCY_PROB_THRESHOLD}`")
        st.markdown(f"**Micro-cube Shape:** `{cfg.MICRO_CUBE_SIZE}^3`")
        st.markdown(f"**Weights File:** `{cfg.MODEL_WEIGHTS_FILENAME}`")
        st.markdown(f"**Device:** `{cfg.INFERENCE_DEVICE}`")
    except Exception as e:
        st.error(str(e))
