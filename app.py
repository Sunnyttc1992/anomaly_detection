import streamlit as st
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from model import Autoencoder, load_preprocessor

# Load model and preprocessing pipeline
input_dim = 16  # Match your training setup
model = Autoencoder(input_dim)
model.load_state_dict(torch.load("autoencoder.pth"))
model.eval()

# Load preprocessor
preprocessor = load_preprocessor()  # See note below

st.title("Pipe Condition Anomaly Detection")

uploaded_file = st.file_uploader("Upload CSV for Prediction", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data", df.head())

    # Preprocess
    X = preprocessor.transform(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        recon = model(X_tensor)
        errors = torch.mean((X_tensor - recon) ** 2, dim=1).numpy()

    # Threshold
    threshold = 0.5888
    df["Reconstruction_Error"] = errors
    df["Anomaly_Flag"] = errors > threshold

    st.write("Anomaly Detection Result", df)
    st.download_button("Download Result CSV", df.to_csv(index=False), "anomaly_result.csv")
