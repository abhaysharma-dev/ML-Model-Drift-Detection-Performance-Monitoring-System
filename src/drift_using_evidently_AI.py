import streamlit as st
import streamlit.components.v1 as components
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset,TargetDriftPreset,ClassificationPreset
from evidently import ColumnMapping
from src.baseline_statistics_math import load_csv
from src.drift_utils import prediction
from src.model_utils import load_model

model = load_model()

@st.cache_data
def html_file(path):
    with open(path,"r",encoding="utf-8") as f:
        html_data = f.read()
    return html_data


def set_column_mapping():
    target = "Churn"
    prediction = "predictions"
    categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']
    numerical_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    return target,prediction,categorical_features,numerical_features


st.set_page_config(layout="wide")
def show_evidently_report(new_data):
    reference = load_csv("data/Telco_Customer_churn.csv")
    current = reference.copy()
    current["TotalCharges"] = current["TotalCharges"]*10
    current["MonthlyCharges"] = current["MonthlyCharges"]*10
    current["predictions"] = prediction(model,current.drop(columns=["customerID","predictions","Churn"]))

    if new_data:
        st.sidebar.markdown("### View Report for :")
        if st.sidebar.button("Data Drift") and new_data:
                show_data_drift(current,reference)
                st.sidebar.success("Successfully loaded the Data Drift")
        if st.sidebar.button("Target Drift") :
                show_target_drift(current,reference)
                st.sidebar.success("Successfully loaded the Target Drift")
        if st.sidebar.button("Classification Drift"):
                show_classification_data(current,reference)
                st.sidebar.success("Successfully loaded the Classification Drift")
    else:
         st.sidebar.error("Upload a CSV file first")


def show_data_drift(current,reference):

    target,prediction,categorical_features, numerical_features = set_column_mapping()
    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.categorical_features = categorical_features
    column_mapping.id = "customerID"
    column_mapping.numerical_features = numerical_features
    data_drift = Report(metrics = [DataDriftPreset()])
    data_drift.run(reference_data=  reference,current_data= current,column_mapping = column_mapping)
    data_drift.save_html("reports/data_drift.html")

    path = "reports/data_drift.html"
    html_data = html_file(path)
    components.html(html_data,height=2000,scrolling=True)

def show_target_drift(current,reference):
    target,prediction,categorical_features, numerical_features = set_column_mapping()

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.categorical_features = categorical_features
    column_mapping.id = "customerID"

    column_mapping.numerical_features = numerical_features
    target_drift = Report(metrics = [TargetDriftPreset()])
    target_drift.run(reference_data= reference,current_data=current,column_mapping = column_mapping)
    target_drift.save_html("reports/target_drift.html")

    path = "reports/target_drift.html"
    html_data = html_file(path)
    components.html(html_data,height=2000,scrolling=True)

def show_classification_data(current,reference):
    target,prediction,categorical_features, numerical_features = set_column_mapping()

    column_mapping = ColumnMapping()
    column_mapping.target = target
    column_mapping.prediction = prediction
    column_mapping.categorical_features = categorical_features
    column_mapping.numerical_features = numerical_features
    column_mapping.id = "customerID"

    classification_drift = Report(metrics = [ClassificationPreset()])
    classification_drift.run(reference_data=reference,current_data=current,column_mapping = column_mapping)
    classification_drift.save_html("reports/classification_drift.html")

    path = "reports/classification_drift.html"
    html_data = html_file(path)
    components.html(html_data,height=2000,scrolling=True)