import streamlit as st
import pickle
import numpy as np
# Streamlit UI
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details to predict heart disease risk.")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, value=120)
chol = st.number_input("Cholesterol Level", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar (>120 mg/dl) (0 = No, 1 = Yes)", [0, 1])
restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

model_path = 'knn_heart_model.pkl'
scaler_path = 'knn_scaler.pkl'

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

loaded_model = load_model(model_path)
loaded_scaler = load_model(scaler_path)

if st.button("Predict"):
    try:
        input_features = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        input_features = loaded_scaler.transform(input_features)

        value = loaded_model.predict(input_features)

        if value == 1:
            st.error("⚠️ High Risk: The patient may have heart disease.")
        else:
            st.success("✅ Low Risk: The patient is unlikely to have heart disease.")
    
        print("predict")
    except Exception as e:
        st.error(f"Error: {e}")
        