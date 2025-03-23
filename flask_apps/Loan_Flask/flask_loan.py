from flask import Flask, render_template, request
import joblib
import pandas as pd

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
    
@app.route('/')
def home():
    return render_template('loan_form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = {
        'person_age': int(request.form['person_age']),
        'person_income': float(request.form['person_income']),
        'person_emp_exp': int(request.form['person_emp_exp']),
        'loan_amnt': float(request.form['loan_amnt']),
        'loan_int_rate': float(request.form['loan_int_rate']),
        'loan_percent_income': float(request.form['loan_percent_income']),
        'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length']),
        'credit_score': int(request.form['credit_score']),
        'person_gender': request.form['person_gender'],
        'person_education': request.form['person_education'],
        'person_home_ownership': request.form['person_home_ownership'],
        'loan_intent': request.form['loan_intent'],
        'previous_loan_defaults_on_file': request.form['previous_loan_defaults_on_file']
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess the data
    # Label encode categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        if col in le.classes_:
            df[col] = le.transform(df[col])
        else:
            print("Label Encoding Error")
        
    # Standardize numerical columns
    numerical_cols = df.select_dtypes(include=['int', 'float']).columns
    df[numerical_cols] = sc.transform(df[numerical_cols])  # Transform entire numerical DataFrame
    
    # Make prediction
    prediction = model.predict(df)
    
    # Render the result template with the prediction
    return render_template('loan_result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)