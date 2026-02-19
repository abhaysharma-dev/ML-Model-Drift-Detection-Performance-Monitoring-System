import streamlit as st
import pandas as pd
from src.baseline_statistics_math import cal_drift
from src.drift_using_evidently_AI import show_evidently_report

st.title("ML Model Drift Detection & Performance Monitoring System")
st.set_page_config("ML Project")

new_data = st.file_uploader("Upload New Dataset(.csv)",type = ["csv"])
tab1= st.sidebar.selectbox("Drift Method",["Baseline_Statistics","Evidently AI"])

if tab1 == "Baseline_Statistics":
    cal_drift(new_data)
    st.sidebar.success("Default BaseLine loaded")
elif tab1 == "Evidently AI":
    st.subheader("Select Drift to view from sidebar")
    st.sidebar.divider()
    show_evidently_report(new_data)

    
