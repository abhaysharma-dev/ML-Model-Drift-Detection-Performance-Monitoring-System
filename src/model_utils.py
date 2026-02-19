import joblib
import streamlit as st

@st.cache_resource
def load_model():
    loaded_pipeline = joblib.load(r"artifacts/model_pipeline.pkl")
    return loaded_pipeline

@st.cache_resource
def load_baseline_stats():
    return joblib.load(r"artifacts/baseline_stats.pkl")

@st.cache_resource
def load_baseline_positive_rate():
    return joblib.load(r"artifacts/baseline_positive_rate.pkl")

def prediction(model,x_test):
    pred = model.predict(x_test)
    return pred