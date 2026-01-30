# ANN-model-on-churn-prediction

A simple Artificial Neural Network (ANN) for customer churn prediction. This repository contains code, notebooks, and artifacts to train, evaluate and use an ANN to predict whether a customer will churn.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ann-model-on-churn-prediction-gcrsvzddyd9u6amnjklzbr.streamlit.app) [![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ann-model-on-churn-prediction-3iktapnyg5x9n4bmjm9yhg.streamlit.app)

## Table of contents
- Overview
- Live demo
- Results
- Dataset
- Model
- Getting started
  - Requirements
  - Install
- Usage
  - Web app (deployed)
  - Run locally (optional)
- Repository structure
- Configuration & hyperparameters
- Acknowledgements
- Contributing
- License
- Contact

## Overview
This project demonstrates how to build and train a feedforward neural network to predict customer churn. It includes data preprocessing, a trained model (Keras .h5), label/one-hot encoders and scaler artifacts, example notebooks, and a Streamlit app for inference.

## Live demo
Open the deployed Streamlit apps:

- Production: https://ann-model-on-churn-prediction-gcrsvzddyd9u6amnjklzbr.streamlit.app
- Alternate/staging: https://ann-model-on-churn-prediction-3iktapnyg5x9n4bmjm9yhg.streamlit.app

Or click the badges at the top of this README to open them.

## Results
(Replace with your actual results after training)
- Accuracy: 0.86
- Precision: 0.79
- Recall: 0.72
- F1-score: 0.75
- AUC-ROC: 0.88

These are example values — run the evaluation scripts or open the notebooks to compute and update real metrics.

## Dataset
The repository contains an example CSV (Churn_Modelling.csv) used for training and demonstration. Typical columns include:
- RowNumber, CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited

Target column: `Exited` (1 = churned, 0 = active)

Note: Do not store PII or sensitive customer data in the repo. Use local or private storage for production datasets.

## Model
This project includes a trained Keras model (model.h5) and supporting preprocessing objects:
- model.h5 — trained ANN model
- scaler.pkl — StandardScaler used for numeric features
- label_encoder_gender.pkl — label encoder for gender
- onehot_geo.pkl / one_hot_geo.pkl — one-hot encoder for geography

The ANN uses a small feedforward architecture with ReLU hidden activations and Sigmoid output for binary classification. Loss: binary cross-entropy; optimizer: Adam.

## Getting started

### Requirements
- Python 3.8+
- pip
- Packages listed in requirements.txt (the repo contains a requirements.txt file)

### Install
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Web app (deployed) — recommended
Open the Streamlit demo(s):

- https://ann-model-on-churn-prediction-gcrsvzddyd9u6amnjklzbr.streamlit.app
- https://ann-model-on-churn-prediction-3iktapnyg5x9n4bmjm9yhg.streamlit.app

These run the inference UI and let users upload input or enter customer features to get churn probability and a prediction.

### Run locally (optional)
This short snippet is optional for contributors who want to run or debug the app locally.

```bash
# optional: run the Streamlit app locally
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Repository structure
Files present in this repository (summary):
- Churn_Modelling.csv
- README.md
- app.py                 # Streamlit app
- experiments.ipynb      # notebook used to build/train the model
- predictions.ipynb      # inference examples in notebook form
- model.h5               # trained Keras model
- scaler.pkl
- scaler_regression.pkl
- label_encoder_gender.pkl
- onehot_geo.pkl
- one_hot_geo.pkl
- regression.py
- salary.ipynb
- salary_prediction_model.h5
- requirements.txt

Update this list if you add/remove files.

## Configuration & hyperparameters
Example hyperparameters (adjust in your training code/notebook):
- hidden_layers: [64, 32]
- activation: ReLU
- dropout: 0.3
- learning_rate: 0.001
- batch_size: 64
- epochs: 50

Record the config used for your best model in `reports/` or a separate `config/` file.

## Acknowledgements
This project was built following Krish Naik's churn prediction tutorial (inspiration). Original tutorial: https://www.youtube.com/c/KrishNaik

If you based code on a specific repository or copied files, please add a proper notice and the original license here.



## License
Add a LICENSE file to the repository. Example: MIT License.

## Contact
Maintainer: monarch-001 (GitHub)
