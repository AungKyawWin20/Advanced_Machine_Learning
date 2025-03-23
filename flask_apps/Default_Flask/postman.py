from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessing objects
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Define predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json
        features = []

        # Define the column order
        column_order = ["person_age", "person_gender", "person_education", "person_income",
                        "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
                        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
                        "credit_score", "previous_loan_defaults_on_file"]

        # Convert input data to feature array
        for col in column_order:
            if col in label_encoders:
                features.append(label_encoders[col].transform([data[col]])[0])
            else:
                features.append(data[col])

        # Scale input features
        features_scaled = scaler.transform([features])

        # Predict using KNN model
        prediction = model.predict(features_scaled)[0]
        
        if prediction == 1:
            return jsonify({"loan_status": "Applicant will likely default on loan"})
        else:
            return jsonify({"loan_status": "Applicant will likely repay loan"})
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
