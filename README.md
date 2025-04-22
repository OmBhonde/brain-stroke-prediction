
# 🧠 Brain Stroke Prediction

A machine learning-based web app that predicts whether a patient is likely to experience a stroke based on their health data.

## 🚀 Project Overview

This project uses patient demographic and health-related features to predict stroke risk using a trained ML model. The app is deployed using Streamlit.

## 📊 Dataset

- **Source**: Kaggle Stroke Prediction Dataset
- Features include: age, gender, hypertension, heart disease, marital status, work type, BMI, glucose level, and smoking status.

## 🧠 ML Model

Trained using RandomForestClassifier, tuned for accuracy. The model was saved using `joblib`.

## 🛠 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/brain-stroke-prediction.git
cd brain-stroke-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app
```bash
streamlit run app.py
```

## 🌐 Live App

👉 [Click here to open Streamlit app](#) (link after deployment)

## 📁 Project Structure

```
brain-stroke-prediction/
├── app.py
├── model.pkl
├── requirements.txt
└── README.md
```

## 🙌 Acknowledgements

- [Kaggle Stroke Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- Streamlit for easy deployment
