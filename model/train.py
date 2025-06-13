import os
import pandas as pd
import numpy as np
import joblib
import kagglehub
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from model import build_autoencoder

# === Download dataset from Kaggle ===
path = kagglehub.dataset_download("muhammadwaqas023/predictive-maintenance-oil-and-gas-pipeline-data")
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
file_path = os.path.join(path, csv_files[0])
df = pd.read_csv(file_path)

# === Preprocessing ===
num_cols = ['Pipe_Size_mm', 'Thickness_mm', 'Max_Pressure_psi',
            'Temperature_C', 'Corrosion_Impact_Percent', 'Time_Years']
cat_cols = ['Material', 'Grade']
X_raw = df[num_cols + cat_cols]

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
])
X = preprocessor.fit_transform(X_raw)
joblib.dump(preprocessor, "preprocessor.joblib")

# === Split ===
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# === Build + Train Autoencoder ===
model = build_autoencoder(input_dim=X.shape[1])
model.fit(X_train, X_train,
          epochs=50,
          batch_size=32,
          validation_data=(X_val, X_val))

# === Save Model ===
model.save("autoencoder_keras.h5")
