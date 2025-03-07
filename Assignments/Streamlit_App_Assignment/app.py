import streamlit as st
import pickle
import numpy as np

# Streamlit UI
st.title("💐 Flower Species Prediction App")
st.write("Enter flower details below:")

# Input fields
sepal_length = st.number_input("Sepal Length", min_value=1.0, max_value=10.0, value=2.5, format="%.2f")
sepal_width = st.number_input("Sepal Width", min_value=1.0, max_value=10.0, value=2.5, format="%.2f")
petal_length = st.number_input("Petal Length", min_value=1.0, max_value=10.0, value=1.0, format="%.2f")
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, value=0.0, format="%.2f")

# Load model
model_path = 'knn_model.pkl'

@st.cache_resource  # Cache the loaded model to improve performance
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

loaded_model = load_model(model_path)

# Define species names (ensure they match model output)
species_names = ['Setosa', 'Versicolor', 'Virginica']

# Prediction button
if st.button("Predict"):
    try:
        # Prepare input features
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
       
        # Get prediction (ensure output is an integer index)
        prediction = loaded_model.predict(input_features)
        
        # Display result
        st.success(f"✅ Predicted Species: {prediction[0]}")

    except Exception as e:
        st.error(f"Error: {e}")