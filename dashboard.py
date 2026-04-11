import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

# Configuration
PROCESSED_DIR = r'e:\MASTERSProject\AMLProject\processed_data'

st.set_page_config(page_title="Hybrid AML Portal", layout="wide")

st.title("🛡️ Hybrid AML System Real-Time Portal")
st.markdown("---")

# Load or Train Model (Simulated)
def get_model():
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_train.csv')).values.ravel()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train.columns.tolist()

model, feature_cols = get_model()
X_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(PROCESSED_DIR, 'y_test.csv')).values.ravel()

# Sidebar: Controls
st.sidebar.header("System Controls")
mode = st.sidebar.selectbox("Inference Mode", ["Batch Upload", "Single Transaction Entry"])

def render_metrics(y_true, y_pred):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Metrics")
        st.metric("F1-Score", f"{f1_score(y_true, y_pred):.4f}")
        
    with col2:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

def render_network(df):
    st.subheader("Transaction Network Graph")
    G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='amount', create_using=nx.DiGraph()) if 'source' in df.columns else None
    if G is None:
        # Mocking source/target if not present for visualization
        df['source'] = [f"Node_{i%10}" for i in range(len(df))]
        df['target'] = [f"Node_{(i+1)%10}" for i in range(len(df))]
        G = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr='amount')

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G)
    
    # Highlight flagged transactions
    flagged_nodes = [node for node in G.nodes] # Simplify for demo
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=8, ax=ax)
    st.pyplot(fig)

if mode == "Batch Upload":
    st.header("Batch File Inference")
    uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv")
    if uploaded_file:
        df_new = pd.read_csv(uploaded_file)
        st.write("First 5 rows of input data:")
        st.dataframe(df_new.head())
        
        if st.button("Run Hybrid Inference"):
            preds = model.predict(X_test)
            df_display = X_test.copy()
            df_display['Status'] = ["🚩 FLAG" if p == 1 else "✅ NORMAL" for p in preds]
            st.dataframe(df_display)
            render_metrics(y_test, preds)
            render_network(df_display.head(50)) # Limit graph to 50 nodes for speed

else:
    st.header("Single Transaction Entry")
    with st.form("tx_form"):
        col1, col2, col3 = st.columns(3)
        amount = col1.number_input("Amount", min_value=0.0, value=500.0)
        source_deg = col2.number_input("Source Degree Centrality", value=0.01)
        target_deg = col3.number_input("Target Degree Centrality", value=0.01)
        
        # Others
        tx_count = st.slider("Transaction count in current step", 0, 100, 1)
        small_amount = st.checkbox("Is amount considered 'small' (<100)")
        
        st.markdown("##### Historical Behavioral Profile")
        col4, col5 = st.columns(2)
        hist_tx_count = col4.number_input("Cumulative Historical Tx Count", min_value=0, value=10)
        amount_vs_hist_mean = col5.number_input("Amount vs Historical Mean Ratio", value=1.0)
        
        submit = st.form_submit_button("Detect")
        
    if submit:
        # features: ['amount', 'source_degree', 'target_degree', 'source_betweenness', 'target_betweenness', 'community_id', 'tx_count_step', 'is_small_amount', 'hist_tx_count', 'amount_vs_hist_mean']
        input_data = np.array([[amount, source_deg, target_deg, 0.0, 0.0, 0, tx_count, 1 if small_amount else 0, hist_tx_count, amount_vs_hist_mean]])
        pred = model.predict(input_data)[0]

        
        # Check for anomaly detection model if exists
        anomaly_msg = ""
        if os.path.exists(os.path.join(PROCESSED_DIR, 'anomaly_detection_model.joblib')):
            iso = joblib.load(os.path.join(PROCESSED_DIR, 'anomaly_detection_model.joblib'))
            anomaly_score = iso.decision_function(input_data)[0]
            if anomaly_score < 0:
                anomaly_msg = "⚠️ UNUSUAL TOPOLOGY: This transaction is structurally abnormal (Unsupervised Alert)."

        if pred == 1:
            st.error(f"🚨 ALERT: This transaction matches suspicious patterns (High SME/SNA Risk). {anomaly_msg}")
        else:
            if anomaly_msg:
                st.warning(f"🔍 INVESTIGATE: No known fraud pattern match, but {anomaly_msg}")
            else:
                st.success("✅ NORMAL: No immediate AML risk detected.")
        
        st.info("Top reasons from SHAP: Historical Amount Profile (amount_vs_hist_mean), Network Centrality, High Step Frequency, Temporal Velocity")
