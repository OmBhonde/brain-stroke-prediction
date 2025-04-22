
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/dphi-official/Datasets/master/Stroke_data.csv')

df = load_data()

# Data Preprocessing
df['gender'].replace('Other', 'Male', inplace=True)
df.dropna(inplace=True)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['ever_married'] = df['ever_married'].map({'No': 0, 'Yes': 1})
df['Residence_type'] = df['Residence_type'].map({'Urban': 0, 'Rural': 1})
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])

X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Streamlit UI
st.title("🧠 Brain Stroke Prediction App (No Model File Needed)")
st.markdown("Enter patient information to predict stroke risk.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 0, 100, 30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.slider("Average Glucose Level", 50.0, 300.0, 100.0)
bmi = st.slider("BMI", 10.0, 60.0, 22.0)
smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])

# One-hot encode manual inputs
input_dict = {
    'gender': 0 if gender == 'Male' else 1,
    'age': age,
    'hypertension': 1 if hypertension == 'Yes' else 0,
    'heart_disease': 1 if heart_disease == 'Yes' else 0,
    'ever_married': 1 if ever_married == 'Yes' else 0,
    'Residence_type': 0 if residence_type == 'Urban' else 1,
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'work_type_Govt_job': 1 if work_type == 'Govt_job' else 0,
    'work_type_Never_worked': 1 if work_type == 'Never_worked' else 0,
    'work_type_Private': 1 if work_type == 'Private' else 0,
    'work_type_Self-employed': 1 if work_type == 'Self-employed' else 0,
    'work_type_children': 1 if work_type == 'Children' else 0,
    'smoking_status_Unknown': 1 if smoking_status == 'Unknown' else 0,
    'smoking_status_formerly smoked': 1 if smoking_status == 'formerly smoked' else 0,
    'smoking_status_never smoked': 1 if smoking_status == 'never smoked' else 0,
    'smoking_status_smokes': 1 if smoking_status == 'smokes' else 0,
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Stroke Risk"):
    prediction = classifier.predict(input_df)[0]
    st.success("⚠️ High Risk of Stroke!" if prediction == 1 else "✅ Low Risk of Stroke.")
