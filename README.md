# Customer Churn Prediction with an Artificial Neural Network

This project trains an artificial neural network (ANN) to estimate whether a bank customer is likely to leave the bank, then provides a small interactive Streamlit application for trying predictions.

The app does not train the model when it starts. It loads a previously trained Keras model and the preprocessing objects saved during training. Keep those files together in the project folder.

## What the project does

The model uses customer information such as credit score, country, gender, age, account tenure, balance, number of products, credit-card ownership, active-member status, and estimated salary. It returns a churn probability between `0` and `1`.

- A probability greater than `0.5` is shown as **likely to churn**.
- A probability less than or equal to `0.5` is shown as **not likely to churn**.

This is a demonstration model, not a financial or operational decision system. Its output should not be treated as a guarantee or as a substitute for validating the model on suitable data.

## Project files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface; prepares the entered customer data and displays a prediction. |
| `model.h5` | Trained Keras neural-network model loaded by the app. |
| `label_encoder_gender.pkl` | Saved scikit-learn encoder that converts gender values to the numeric representation used in training. |
| `onehot_encoder_geo.pkl` | Saved one-hot encoder that converts geography into model input columns. |
| `scaler.pkl` | Saved `StandardScaler` used to normalize model input features. |
| `Churn_Modelling.csv` | Customer churn dataset used by the training and exploration notebooks. |
| `experiments.ipynb` | Main preprocessing and ANN training notebook. It trains the model and saves `model.h5`, the encoders, and the scaler. |
| `hyperparametertuningann.ipynb` | Notebook for trying different ANN layer sizes and epoch counts with grid search. |
| `prediction.ipynb` | Notebook example of loading the saved artifacts and predicting churn for one customer. |
| `salaryregression.ipynb` | Separate regression experiment; it is not used by the Streamlit churn app. |
| `requirements.txt` | Python package requirements for the project. |

## Requirements

- Windows, macOS, or Linux
- Python 3.13. The requirements pin TensorFlow and TensorBoard to `2.20.0` and require scikit-learn `1.6` or newer (below `2.0`), matching the Python 3.13 environment used for this project.
- The project files listed above. In particular, `app.py`, `model.h5`, and all three `.pkl` files must be in the same working directory when starting the app.

## Install and run on Windows PowerShell

Open PowerShell in the project folder (the folder containing `app.py`), then run:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation, enable it for just the current PowerShell process and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The Streamlit command prints a local URL (usually `http://localhost:8501`). Open that URL in a browser. Stop the server by pressing `Ctrl+C` in the terminal.

### Run without activating the environment

Activation is optional. After creating the virtual environment and installing the packages, you can invoke its Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Run from Command Prompt

In Command Prompt, move to the project folder and use:

```bat
py -3.13 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Using the application

1. Start the app using the command above.
2. Select a geography and gender, and enter or select the remaining customer details.
3. Read the displayed **Churn Probability** and the accompanying classification.

The form controls define convenient ranges for some values, but numeric inputs such as balance, credit score, and estimated salary should still be entered using sensible values for the dataset.

## How prediction works

`app.py` performs these steps whenever Streamlit runs the script:

1. Loads `model.h5` with TensorFlow/Keras and loads the gender encoder, geography encoder, and scaler from their `.pkl` files.
2. Collects the user's customer details from Streamlit widgets.
3. Encodes gender with the saved `LabelEncoder` and geography with the saved `OneHotEncoder`.
4. Combines the encoded categorical values with the numeric customer features in the feature layout expected by the model.
5. Applies the saved `StandardScaler`. The same scaler fitted on training data must be used for prediction.
6. Passes the scaled values to the model. The output layer uses a sigmoid activation, so its output is interpreted as a churn probability.
7. Displays the probability and applies the app's `0.5` decision threshold.

The encoders, scaler, and model are a matched set. If you retrain or change preprocessing, regenerate and use all corresponding artifacts together; mixing artifacts from different training runs can produce invalid predictions. The supplied `.pkl` files were serialized with scikit-learn `1.5.0`, so scikit-learn may warn that they were loaded under a different version. The app smoke test succeeds in the current Python 3.13 environment, but to remove the version mismatch and ensure the saved preprocessing objects match your installed scikit-learn, rerun the training notebook and use the newly generated artifacts.

## Training or exploring the model

The Streamlit app only runs inference. To repeat the main training workflow:

1. Install the project requirements and a Jupyter notebook interface if one is not already available:

   ```powershell
   python -m pip install notebook
   python -m notebook
   ```

2. Open `experiments.ipynb` from the project folder and run its cells in order.
3. The notebook reads `Churn_Modelling.csv`, removes the row number, customer ID, and surname, encodes gender and geography, splits the data into training and test sets, and scales the features.
4. It trains a dense ANN with two ReLU hidden layers and a sigmoid output layer for binary classification. Early stopping monitors validation loss, and TensorBoard logging is configured.
5. The notebook saves the trained model and preprocessing artifacts to the project folder. Use the resulting files with `app.py`.

`hyperparametertuningann.ipynb` is optional and can take considerably longer because it evaluates multiple layer sizes and epoch counts with cross-validation. Review its final cells and results before using it: the notebook's grid search is an experiment and does not by itself guarantee that the app's `model.h5` has been replaced with a tuned model.

`prediction.ipynb` demonstrates a single prediction outside the web app. It requires the same model and preprocessing artifacts as `app.py`.

## Common issues

- **A model or pickle file is missing:** Run the training notebook or restore the corresponding artifact, and ensure the files are beside `app.py`.
- **The app cannot find files:** Start Streamlit from the project folder, not from a different working directory.
- **TensorFlow installation fails:** Confirm you are using Python 3.13 and reinstall `requirements.txt` in the project virtual environment. TensorFlow 2.20.0 is the version selected for this project.
- **scikit-learn warns about a saved estimator version:** The existing `.pkl` files were saved with scikit-learn 1.5.0. Rerun `experiments.ipynb` in the current environment to regenerate them.
- **A new geography or gender value causes an error:** The input must be supported by the saved encoders. If the training data or categories change, retrain and regenerate the matched artifacts.
- **The app displays a probability but the result is unexpected:** Confirm the model, encoders, and scaler came from the same preprocessing/training run and that values were entered in the expected units.

## Useful commands

```powershell
# See the Python version being used
python --version

# Install or update packages from the project requirements
python -m pip install -r requirements.txt

# Start the Streamlit app
python -m streamlit run app.py

# Stop the running app
# Press Ctrl+C in the terminal
```
