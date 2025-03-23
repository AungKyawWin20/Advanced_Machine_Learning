from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessing objects
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")
label_encoders = joblib.load("label_encoders.joblib")

# Define the column order
column_order = ["person_age", "person_gender", "person_education", "person_income",
                "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
                "credit_score", "previous_loan_defaults_on_file"]

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            # Collect user inputs from form
            user_input = {}
            for col in column_order:
                user_input[col] = request.form[col]

            # Convert categorical variables using LabelEncoders
            features = []
            for col in column_order:
                if col in label_encoders:
                    features.append(label_encoders[col].transform([user_input[col]])[0])
                else:
                    features.append(float(user_input[col]))  # Convert numeric inputs

            # Scale input features
            features_scaled = scaler.transform([features])

            # Make prediction
            prediction = model.predict(features_scaled)[0]
            result_text = "Approved" if prediction == 1 else "Rejected"

            return render_template("result.html", prediction=result_text)
        
        except Exception as e:
            return render_template("result.html", prediction=f"Error: {str(e)}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
