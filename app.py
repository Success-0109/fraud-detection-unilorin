import streamlit as st
import pandas as pd
import numpy as np
import pickle
from streamlit_option_menu import option_menu 

# --- PAGE CONFIG ---
st.set_page_config(page_title="FRAUD_DETECTION_SYS", layout="wide", page_icon="🛡️")

# --- CUSTOM CSS FOR DARK DASHBOARD LOOK ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    .metric-card {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    h1, h2, h3 { color: #58a6ff; font-family: 'Inter', sans-serif; }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #238636;
        color: white;
        border: none;
        height: 3em;
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #2ea043; border: none; }
    .fraud-box { background-color: #442d2d; border: 1px solid #f85149; padding: 20px; border-radius: 10px; color: #f85149; }
    .legit-box { background-color: #1c2d22; border: 1px solid #3fb950; padding: 20px; border-radius: 10px; color: #3fb950; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = pickle.load(open('rf_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    samples = pickle.load(open('samples.pkl', 'rb'))
    return model, scaler, samples

model, scaler, samples = load_resources()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🛡️ FRAUD_SYS")
    st.write("UNIVERSITY OF ILORIN")
    st.markdown("---")
    selected = option_menu(
        menu_title="FINAL YEAR PROJECT",
        options=["Dashboard", "Transaction Analysis", "Model Comparison", "Methodology"],
        icons=["speedometer2", "activity", "bar-chart", "book"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5!important", "background-color": "#161b22"},
            "icon": {"color": "#58a6ff", "font-size": "20px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#30363d", "color": "white"},
            "nav-link-selected": {"background-color": "#094771"},
        }
    )

# --- DASHBOARD PAGE ---
if selected == "Dashboard":
    st.title("System Overview")
    st.write("Live telemetry and model performance metrics.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><h4>Model Accuracy</h4><h2 style="color:#58a6ff">99.95%</h2><p>Random Forest Classifier</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h4>Precision / Recall</h4><h2 style="color:#58a6ff">84.5% / 83.7%</h2><p>SMOTE Balanced</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h4>Dataset Size</h4><h2 style="color:#58a6ff">284,807</h2><p>Transactions Evaluated</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><h4>Fraud Baseline</h4><h2 style="color:#f85149">0.172%</h2><p>492 Total Cases</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.subheader("Confusion Matrix Analysis")
        st.image('figure_4_4_confusion_matrices.png', use_container_width=True)
    with right_col:
        st.subheader("Live Feed Simulation")
        st.info("System Status: Online")
        st.success("Log: 56,857 Legit detected")
        st.error("Log: 82 Fraud intercepted")
        st.warning("Log: Class Imbalance handled via SMOTE")

# --- TRANSACTION ANALYSIS PAGE (NEW DYNAMIC VERSION) ---
elif selected == "Transaction Analysis":
    st.title("Transaction Analysis Engine")
    st.write("Run individual transactions through the classification model.")
    
    # Initialize session state for values
    if 'current_vals' not in st.session_state:
        st.session_state.current_vals = samples['legit'][0]

    st.subheader("Select Sample to Load")
    col_select1, col_select2 = st.columns(2)
    
    fraud_choice = col_select1.selectbox("🚨 Choose a Fraud Case signature", options=[1, 2, 3, 4, 5])
    if col_select1.button("Load Selected Fraud Signature"):
        st.session_state.current_vals = samples['fraud'][fraud_choice - 1]
        st.rerun() 

    legit_choice = col_select2.selectbox("✅ Choose a Legit Case signature", options=[1, 2, 3, 4, 5])
    if col_select2.button("Load Selected Legit Signature"):
        st.session_state.current_vals = samples['legit'][legit_choice - 1]
        st.rerun()

    st.markdown("---")
    col_input, col_result = st.columns([2, 1.2])
    
    with col_input:
        st.subheader("Feature Inputs")
        t_col, a_col = st.columns(2)
        tm = t_col.number_input("Time (seconds)", value=float(st.session_state.current_vals['Time']))
        amt = a_col.number_input("Amount (€)", value=float(st.session_state.current_vals['Amount']))
        
        st.write("PCA-transformed features (V1-V28)")
        v_inputs = {}
        v_cols = st.columns(4)
        for i in range(1, 29):
            idx = (i-1) % 4
            v_inputs[f'V{i}'] = v_cols[idx].number_input(f"V{i}", value=float(st.session_state.current_vals[f'V{i}']), format="%.4f")

    with col_result:
        st.subheader("Evaluation Verdict")
        if st.button("🚀 RUN CLASSIFICATION"):
            columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
            input_row = [tm] + [v_inputs[f'V{i}' ] for i in range(1, 29)] + [amt]
            input_df = pd.DataFrame([input_row], columns=columns)
            input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
            
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]
            
            if pred == 1:
                st.markdown(f'<div class="fraud-box"><h3>🚨 CRITICAL: FRAUD DETECTED</h3><p>Model Confidence: {prob*100:.2f}%</p></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="legit-box"><h3>✅ AUTHORIZED: LEGITIMATE</h3><p>Model Confidence: {(1-prob)*100:.2f}%</p></div>', unsafe_allow_html=True)
            
            st.markdown("---")
            st.subheader("Decision Impact")
            st.write("Feature Importance contributes to this decision:")
            st.image('figure_4_6_feature_importance.png', use_container_width=True)

# --- MODEL COMPARISON PAGE ---
elif selected == "Model Comparison":
    st.title("Performance Metrics")
    st.write("Side-by-side evaluation of core classifiers.")
    st.image('figure_4_5_roc_curves.png', use_container_width=True)
    st.markdown("---")
    st.subheader("Top Indicator Features")
    st.image('figure_4_6_feature_importance.png', use_container_width=True)

# --- METHODOLOGY PAGE ---
elif selected == "Methodology":
    st.title("Project Methodology")
    st.subheader("1. Dataset Description")
    st.write("The dataset contains transactions made by credit cards in September 2013 by European cardholders. It contains 492 frauds out of 284,807 transactions.")
    
    st.subheader("2. Data Preprocessing & SMOTE")
    st.write("SMOTE was applied to correct the imbalance ratio, generating synthetic examples for the minority class.")
    
    st.subheader("3. Random Forest Algorithm")
    st.write("Random Forest was chosen for its ensemble learning capability and robustness against overfitting.")

# --- FOOTER ---
st.markdown("---")
st.caption("© 2026 UNIVERSITY OF ILORIN | FINAL YEAR PROJECT | DEPARTMENT OF COMPUTER SCIENCE")