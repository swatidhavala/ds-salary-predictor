# 💼 DS Salary Predictor

An interactive Streamlit app that predicts Data Science salaries 
using a Weighted Ensemble of XGBoost, LightGBM and Gradient Boosting.

## 🚀 Live App
https://ds-salary-predictor-cvsgodagh6ngu3em7bresp.streamlit.app

## 📊 Features
- **Predict** — Enter your profile and get an estimated salary
- **EDA** — Explore salary trends by experience, work model, company size
- **Model Performance** — R² metrics, residual plots, feature importance
- **SHAP** — Understand why the model made each prediction

## 🤖 Model
- XGBoost + LightGBM + Gradient Boosting
- Weighted ensemble (weights assigned by R² score)
- Ensemble R²: 0.5143 | Avg Error: ~$63,018

## 📁 Project Structure
```
ds_salary_predictor/
├── data/
│   └── data_science_salaries.csv
├── regression.ipynb
├── app.py
├── requirements.txt
└── README.md
```

## 📦 Dataset
Download from Kaggle:
```bash
kaggle datasets download -d sazidthe1/data-science-salaries -p ./data --unzip
```

## ⚠️ Known Limitations
- R² ~0.51 — salary data has high natural variation across countries
- Executive-level predictions less reliable (only 254 samples)
- No features for skills, education or city-level data

## 🛠️ Built With
- Python · Streamlit · XGBoost · LightGBM · SHAP · Scikit-learn
