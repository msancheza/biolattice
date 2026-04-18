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

st.title("Bio-Lattice Dashboard")
st.markdown("Interface for the research pipeline and feature classification.")
st.divider()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_script(script_name):
    try:
        result = subprocess.run(["python", script_name], cwd=BASE_DIR, capture_output=True, text=True)
        return result.stdout, result.stderr
    except Exception as e:
        return "", str(e)

tabs = st.tabs([
    "ℹ️ Project Overview",
    "1. Data Extraction",
    "2. Model Training",
    "3. Evaluation Metrics",
    "4. Prediction",
    "5. Visualization 🔬",
    "6. Config Guide",
])
tab_about, tab1, tab2, tab3, tab4, tab5, tab6 = tabs

with tab_about:
    st.markdown("## 🧬 Bio-Lattice: Documentation & Overview")
    st.markdown("""
    This project focuses on the conversion of 4D MRI volumes (DICOM) into compact representation tensors known as **Micro-Cubes**. 
    The configuration involves a 4-channel $64^3$ structure designed to isolate specific imaging characteristics:

    *   **C1 (Anatomy):** Structural post-contrast average.
    *   **C2 (Heterogeneity):** Local variance map for texture analysis.
    *   **C3 (Kinetics):** Log-compressed enhancement ratio.
    *   **C4 (Vascularity):** Isolated signals of peak-to-average focal realce.
    
    ### ⚙️ Pipeline Lifecycle:
    1. **Data Extraction:** DICOM series selection, volume registration, and ROI resampling.
    2. **Model Training:** Training logic for a 3D-ResNet using tensors and physical metadata.
    3. **Evaluation:** Quantitative assessment using ROC/AUC metrics on separate validation cohorts.
    4. **Prediction:** Probability estimation for individual research cases.
    5. **Visualization:** Channel decomposition and attention map (Grad-CAM) analysis.
    
    > **⚠️ Research Prototype:**
    > This interface and the underlying models are part of a research implementation. These tools are NOT for clinical or diagnostic use.
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
    st.markdown("### Clinical Validation & ROC Analysis")
    
    def parse_metrics_file(filepath):
        """Parses a Bio-Lattice metrics log file into a dict."""
        import ast
        data = {"roc_curve": {"fpr": [], "tpr": []}, "confusion": {}}
        try:
            with open(filepath, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if "roc_auc:" in line: data["auc"] = float(line.split(":")[1].strip())
                    if "accuracy:" in line and "best" not in line.lower(): data["accuracy"] = float(line.split(":")[1].strip())
                    if "sensitivity:" in line and "best" not in line.lower(): data["sensitivity"] = float(line.split(":")[1].strip())
                    if "specificity:" in line and "best" not in line.lower(): data["specificity"] = float(line.split(":")[1].strip())
                    if "configured_threshold:" in line: data["configured_threshold"] = float(line.split(":")[1].strip())
                    if "tn:" in line: data["confusion"]["tn"] = int(line.split(":")[1].strip())
                    if "fp:" in line: data["confusion"]["fp"] = int(line.split(":")[1].strip())
                    if "fn:" in line: data["confusion"]["fn"] = int(line.split(":")[1].strip())
                    if "tp:" in line: data["confusion"]["tp"] = int(line.split(":")[1].strip())
                    if "fpr:" in line: data["roc_curve"]["fpr"] = ast.literal_eval(line.split("fpr:")[1].strip())
                    if "tpr:" in line: data["roc_curve"]["tpr"] = ast.literal_eval(line.split("tpr:")[1].strip())
            
            # Calculate F1-Score (Rigor)
            tp, fp, fn = data["confusion"].get("tp", 0), data["confusion"].get("fp", 0), data["confusion"].get("fn", 0)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = data.get("sensitivity", 0)
            data["f1_score"] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return data
        except Exception as e:
            return {"error": str(e)}

    col_btn_run, col_btn_load = st.columns(2)
    
    res = None
    if col_btn_run.button("Run Fresh Inference & Plot", use_container_width=True):
        with st.spinner("Evaluating validation set..."):
            try:
                if BASE_DIR not in sys.path: sys.path.append(BASE_DIR)
                import predict, importlib
                importlib.reload(predict)
                res = predict.evaluate_dataset()
            except Exception as e:
                st.error(f"Evaluation error: {e}")
                
    if col_btn_load.button("Load Latest Results from Files", use_container_width=True):
        log_dir = os.path.join(BASE_DIR, "dashboard", "training_logs")
        metrics_files = [f for f in os.listdir(log_dir) if f.startswith("metrics_run") and f.endswith(".txt")]
        if metrics_files:
            latest_file = os.path.join(log_dir, sorted(metrics_files)[-1])
            res = parse_metrics_file(latest_file)
            if "error" in res: st.error(res["error"])
            else: st.success(f"Loaded: {os.path.basename(latest_file)}")
        else:
            st.warning("No metrics logs found.")

    if res and "error" not in res:
        st.markdown("#### Performance Metrics")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("ROC AUC", f"{res['auc']:.4f}")
        c2.metric("Accuracy", f"{res['accuracy']:.2%}")
        c3.metric("Sensitivity", f"{res['sensitivity']:.2%}")
        c4.metric("Specificity", f"{res['specificity']:.2%}")
        c5.metric("F1-Score", f"{res.get('f1_score', 0):.4f}")
        
        st.divider()
        
        col_roc, col_cm = st.columns([2, 1])
        
        with col_roc:
            st.markdown("#### ROC Curve (Model Performance)")
            curve = res.get("roc_curve", {})
            if curve.get("fpr") and curve.get("tpr"):
                import plotly.graph_objects as go
                fig_roc = go.Figure()
                # Diagonal Chance
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Chance', line=dict(dash='dash', color='gray')))
                # Bio-Lattice Model
                fig_roc.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode='lines', name='Bio-Lattice v2.6', line=dict(color='#FF4B4B', width=3)))
                
                fig_roc.update_layout(
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate',
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400,
                    showlegend=True,
                    template="plotly_white"
                )
                st.plotly_chart(fig_roc, use_container_width=True)
                st.caption("A curve above the diagonal suggests predictive signal better than random guessing.")
            else:
                st.warning("ROC data points not available for this run.")
                
        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm = res["confusion"]
            st.write(f"**TN:** {cm.get('tn', 0)} | **FP:** {cm.get('fp', 0)}")
            st.write(f"**FN:** {cm.get('fn', 0)} | **TP:** {cm.get('tp', 0)}")
            st.caption(f"Threshold: {res.get('configured_threshold', 0):.2f}")

with tab4:
    st.markdown("### Model Prediction")
    p_id_inf = st.text_input("Patient ID for Prediction:", placeholder="Breast_MRI_002", key="inf_pid")
    if st.button("Generate Prediction", type="primary", use_container_width=True):
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
    st.markdown("### 🔬 4-Channel Decomposition")
    p_id_viz = st.text_input("Patient ID for Visualization:", placeholder="Breast_MRI_400", key="viz_pid")
    if st.button("Generate Visualization", type="primary", use_container_width=True):
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
