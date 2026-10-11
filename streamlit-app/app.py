"""
Heart Disease Prediction - Student Demo App
----------------------------------------------------------
A small Streamlit web app that lets students interact with the trained
Support Vector Classifier from the "Heart Disease Prediction" notebook, without needing to install Python or run Jupyter.

It loads the same svc_trained_model.pkl produced by that notebook, and
recreates the exact same LabelEncoders (fit on the same fixed category
lists) used during training, so predictions match the notebook exactly.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# ---------------------------------------------------------------------------
# Load the trained model (same file the notebook produced)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def build_label_encoders():
    """
    Recreates the exact same LabelEncoders used in the notebook.
    LabelEncoder assigns codes based on the sorted order of the category
    list, so fitting on the same fixed lists reproduces the same codes
    deterministically - no need to pickle/ship the encoders separately.
    """
    encoders = {}
    encoders["Gender"] = LabelEncoder().fit(["Male", "Female"])
    encoders["ChestPain"] = LabelEncoder().fit(["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    encoders["FastingBS"] = LabelEncoder().fit(["True", "False"])
    encoders["RestECG"] = LabelEncoder().fit(["Normal", "ST-T wave", "LVH"])
    return encoders

model = load_model()
encoders = build_label_encoders()

# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------

st.title("❤️ Heart Disease Predictor")
st.markdown(
    """
    This is a teaching demo built from the **Heart Disease Prediction**
    notebook. Pick a patient's details below and the trained Support Vector
    Classifier (SVC) will predict whether they have heart disease.

    *Model accuracy on held-out test data: **varies**.*
    """
)

st.divider()

col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave", "LVH"])

if st.button("Predict Heart Disease", type="primary", use_container_width=True):
    user_input = pd.DataFrame({
        "Gender": [gender],
        "ChestPain": [chest_pain],
        "FastingBS": [fasting_bs],
        "RestECG": [rest_ecg],
    })

    encoded = pd.DataFrame({
        "Gender": encoders["Gender"].transform(user_input["Gender"]),
        "ChestPain": encoders["ChestPain"].transform(user_input["ChestPain"]),
        "FastingBS": encoders["FastingBS"].transform(user_input["FastingBS"]),
        "RestECG": encoders["RestECG"].transform(user_input["RestECG"]),
    })

    prediction = model.predict(encoded)[0]

    st.divider()
    if prediction == 1:
        st.error("### ❌ HEART DISEASE DETECTED")
    else:
        st.success("### ✅ NO HEART DISEASE")

    with st.expander("See the encoded feature vector sent to the model"):
        st.dataframe(encoded, hide_index=True)

st.divider()
st.caption(
    "Built for teaching purposes from a simplified Heart Disease dataset. "
    "Predictions reflect patterns in this toy dataset, not a rigorous "
    "medical diagnosis."
)
