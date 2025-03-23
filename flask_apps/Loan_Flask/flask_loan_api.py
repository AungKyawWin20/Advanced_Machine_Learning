from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load pre-trained model
with open(r'C:\Users\akw97\Desktop\Parami Courses\Advanced_Machine_Learning\flask_apps\Loan_Flask\loan_pred_model.joblib', 'rb') as f:
    model = joblib.load(f)
    
# Load Label Encoder
with open(r'C:\Users\akw97\Desktop\Parami Courses\Advanced_Machine_Learning\flask_apps\Loan_Flask\label_encoder.joblib', 'rb') as f:
    le = joblib.load(f)
    
# Load Standard Scaler
with open(r'C:\Users\akw97\Desktop\Parami Courses\Advanced_Machine_Learning\flask_apps\Loan_Flask\standard_scaler.joblib', 'rb') as f:
    sc = joblib.load(f)

@app.route('/predict_postman', methods=['POST'])
def predict_postman():
    # Get JSON data from Postman
    data = request.get_json()
    
    # Extract features from JSON payload
    input_data = {
        'person_age': int(data['person_age']),
        'person_income': float(data['person_income']),
        'person_emp_exp': int(data['person_emp_exp']),
        'loan_amnt': float(data['loan_amnt']),
        'loan_int_rate': float(data['loan_int_rate']),
        'loan_percent_income': float(data['loan_percent_income']),
        'cb_person_cred_hist_length': int(data['cb_person_cred_hist_length']),
        'credit_score': int(data['credit_score']),
        'person_gender': data['person_gender'],
        'person_education': data['person_education'],
        'person_home_ownership': data['person_home_ownership'],
        'loan_intent': data['loan_intent'],
        'previous_loan_defaults_on_file': data['previous_loan_defaults_on_file']
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([input_data])
    
    # Preprocess the data
    # Label encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if col in le.classes_:
            df[col] = le.transform(df[col])
        else:
            return jsonify({
                'error': f"Unseen category '{df[col].iloc[0]}' in column '{col}'."
            }), 400
    
    # Standardize numerical columns
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    df[numerical_cols] = sc.transform(df[numerical_cols])  # Transform entire numerical DataFrame
    
    # Make prediction
    prediction = model.predict(df)
    
    # Return prediction as JSON response
    return jsonify({
        'prediction': int(prediction[0]),
        'message': 'Loan status predicted successfully.'
    })

if __name__ == '__main__':
    app.run(debug=True)