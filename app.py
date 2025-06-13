import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from model import build_autoencoder

# Load model & preprocessor
model = tf.keras.models.load_model("autoencoder_keras.h5")
preprocessor = joblib.load("preprocessor.joblib")
threshold = 0.5888  # Adjust based on training results

st.title("Pipeline Anomaly Detection")

uploaded_file = st.file_uploader("Upload pipeline dataset", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data Preview", df.head())

    # Preprocess
    X = preprocessor.transform(df)
    preds = model.predict(X)
    errors = np.mean((X - preds) ** 2, axis=1)

    # Flag anomalies
    df["Reconstruction_Error"] = errors
    df["Anomaly_Flag"] = errors > threshold
    st.write("Prediction Results", df)

    # Download
    st.download_button("Download Results", df.to_csv(index=False), "anomaly_results.csv")
    
