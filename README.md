# Disease Detection Application using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-LogReg-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Streamlit-based machine learning web application that predicts the likelihood of four different diseases from patient questionnaire data. Built as a class project for **BBM467 — Data Intensive Applications** at Hacettepe University.

## Overview

Some diseases are difficult to diagnose because they are rare or share overlapping symptoms. This project trains a multi-class classifier on patient questionnaire data and exposes it as a simple web form so a clinician (or any user) can enter feature values and obtain ranked disease probabilities.

The training pipeline uses **chi-squared feature selection (SelectKBest, k=12)** followed by **Logistic Regression** and achieves **~97% accuracy** on the held-out test set. The trained model is serialized to `classifier.pkl` and served by a Streamlit front-end.

## Dataset

- **Source**: `lib/sdsp_patients.xlsx` (400 samples, 51 columns)
- **Target**: `Disease` column with 4 classes (Disease_1 … Disease_4)
- **Description**: See `lib/SDSP_description.pdf` for the full feature dictionary

Class distribution: 244 / 78 / 52 / 26 samples across the four disease classes.

## Tech Stack

- **Language**: Python 3.8+
- **ML**: scikit-learn (LogisticRegression, SelectKBest with chi²)
- **Data**: pandas, numpy, openpyxl (Excel reading)
- **Visualization**: matplotlib, seaborn
- **Web UI**: Streamlit
- **Serialization**: pickle

## Project Structure

```
.
├── src/
│   ├── main.py           # Data preparation, feature selection, model training
│   ├── app.py            # Streamlit web UI
│   ├── SessionState.py   # Streamlit session-state helper
│   └── classifier.pkl    # Pre-trained Logistic Regression model
├── lib/
│   ├── sdsp_patients.xlsx     # Patient questionnaire dataset
│   └── SDSP_description.pdf   # Feature dictionary
├── report .ipynb         # Full project report (data understanding → evaluation)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/Msubasi1/Disease-Detection-Application-using-Machine-Learning.git
cd Disease-Detection-Application-using-Machine-Learning
python3 -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Run the Streamlit web app

```bash
cd src
streamlit run app.py
```

The app opens at `http://localhost:8501`. Fill the form (Gender, Feature_2 … Feature_50) and click **Predict** to see the top-3 disease probabilities.

> **Note**: `app.py` uses the legacy `streamlit.caching` module which was removed in Streamlit ≥ 1.18. The pinned version in `requirements.txt` is compatible.

### Re-train the model

```bash
cd src
python main.py
```

This regenerates `classifier.pkl`. The training script also plots accuracy vs. number-of-features for the chi²-based feature-selection sweep (k = 12, 16, 20, 24, 30, 36, 42, 48).

### View the report notebook

```bash
jupyter notebook "report .ipynb"
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | **~97%** |
| Best k (SelectKBest) | 12 features |
| Model | Logistic Regression (lbfgs, max_iter=4000) |

The report notebook contains the full evaluation: confusion matrix, classification report, and per-class F1 scores.

## Authors

- **Muhammet Subaşı** ([@Msubasi1](https://github.com/Msubasi1)) — 21627601
- **Harun Burkuk** — 21627045

## License

Released under the [MIT License](LICENSE).
