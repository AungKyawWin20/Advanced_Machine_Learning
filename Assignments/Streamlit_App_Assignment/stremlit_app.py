import streamlit as st
import pickle
import numpy as np

# Streamlit UI
st.title("💐 Flower Species Prediction App")
st.write("Enter flower details")

# Input fields
sepal_length = float(st.number_input("sepal_length", min_value=1.0, max_value=10.0, value=2.5, format="%.2f"))
sepal_width = float(st.number_input("sepal_width", min_value=1.0, max_value=10.0, value=2.5, format="%.2f"))
petal_length = float(st.number_input("petal_length", min_value=1.0, max_value=10.0, value=1.0, format="%.2f"))
petal_width = float(st.number_input("petal_width", min_value=0.0, max_value=10.0, value=0.0, format="%.2f"))

model_path = 'knn_model.pkl'

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

loaded_model = load_model(model_path)

species_names = ['Setosa', 'Versicolor', 'Virginica', 'Iris-setosa']

if st.button("Predict"):
    try:
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = loaded_model.predict(input_features)
        species = species_names[prediction[0]]
        st.success(f"✅ Predicted Species: {species}")
    except Exception as e:
        st.error(f"Error: {e}")