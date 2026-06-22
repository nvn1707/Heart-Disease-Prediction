# 🫀 Heart Disease Prediction System

A machine learning web application that predicts heart disease risk based on patient medical data.

## 🚀 Live Demo
[Click here to view the app](https://heart-disease-prediction-f3pofxrsoqmxlxupa65dc6.streamlit.app/)

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| Logistic Regression | 79.51% |
| Random Forest | **98.54%** ✅ |
| XGBoost | 98.54% |

## 🔍 Features
- Predicts heart disease risk from 13 medical parameters
- Trained on 1025 patient records
- Compares 3 ML models and selects the best
- Real-time prediction with confidence score
- Interactive web interface built with Streamlit

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- Joblib

## ⚙️ Run Locally
```bash
git clone https://github.com/nvn1707/Heart-Disease-Prediction
cd Heart-Disease-Prediction
pip install -r requirements.txt
python model/train.py
streamlit run app/app.py
```

## 👤 Author
Naveen B