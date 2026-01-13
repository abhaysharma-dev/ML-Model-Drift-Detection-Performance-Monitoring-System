import streamlit as st
import pandas as pd
from drift_utils import feature_drift,prediction_drift,compute_status
from model_utils import load_baseline_stats,load_model

st.title("ML Model Drift Detection & Performance Monitoring System")
st.set_page_config("ML Project")
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

new_data = st.file_uploader("Upload New Dataset(.csv)",type = ["csv"])

if new_data is not None:
    new_data_df =load_csv("Telco-Customer-Churn.csv")
    new_data_df["MonthlyCharges"] = new_data_df["MonthlyCharges"] * 1.6
    new_data_df.drop(columns = ['customerID','Churn'])
    pipeline = load_model()
    if pipeline is not None:
        st.success("Model Pipeline loaded Successfully!")
    else:
        st.error("Pipeline Not Imported")

if "features" not in st.session_state:
    st.session_state.features = False
if  st.button("Calculate Feature Drift"):
    st.session_state.features= True
    if new_data:
        if "feature_drift_result" not in st.session_state:
            st.session_state.feature_drift_result = feature_drift(new_data_df)
        st.write(st.session_state.feature_drift_result)
        st.dataframe(st.session_state.feature_drift_result)
    else:
        st.error("Upload The csv file first")

if "pred" not in st.session_state:
    st.session_state.pred = False
if st.button("Calculate Prediction Drift"):
    st.session_state.pred = True
    if  st.session_state.features:
        if "pred_drift" not in st.session_state:
            st.session_state.pred_drift = prediction_drift(new_data_df)
        st.write(st.session_state.pred_drift)
    else:
        st.error("Calculate Feature Drift first.")

if st.button("Check Status"):
    if st.session_state.features and st.session_state.pred:
        status = compute_status(st.session_state.pred_drift["drift_detected"],st.session_state.feature_drift_result)
        if status == "STABLE":
            st.success("🟢 MODEL STATUS: STABLE")
        elif status == "MONITOR":
            st.warning("🟡 MODEL STATUS: MONITOR")
        else:
            st.error("🔴 MODEL STATUS: HIGH RISK")

    else:
        st.error("Calculate Feature Drift and Prediction Drift First!")
