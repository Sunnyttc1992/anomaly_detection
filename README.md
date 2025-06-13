This repository demonstrates how deep learning can be applied to anomaly detection.
The method used is an autoencoder neural network.
The training dataset was obtained from Kaggle and contains 1000 rows and 11 columns.

## Training the model

Two files are required for the Streamlit app to run:

* `autoencoder_keras.h5` – the trained autoencoder model
* `preprocessor.joblib` – the fitted `ColumnTransformer` used for preprocessing

These files are not stored in the repository. To generate them yourself:

1. Install the dependencies

   ```bash
   pip install -r model/requirements.txt
   ```

2. Ensure your Kaggle API credentials are configured so that `kagglehub` can download the training dataset.
   Follow the [Kaggle API instructions](https://github.com/Kaggle/kaggle-api#api-credentials) to create the `~/.kaggle/kaggle.json` file.

3. From the repository root, run the training script

   ```bash
   python model/train.py
   ```

   This will download the dataset, train the autoencoder and save `autoencoder_keras.h5`
   and `preprocessor.joblib` in the repository root.

## Running the application

After training, start the Streamlit app with

```bash
streamlit run app.py
```

The application looks for `autoencoder_keras.h5` and `preprocessor.joblib` in the
same directory as `app.py` and will fail to start if they are missing.
