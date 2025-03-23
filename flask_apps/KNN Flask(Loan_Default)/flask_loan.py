from flask import Flask, render_template, request,jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load pre-trained model
with open('loan_pred_model.joblib', 'rb') as f:
    model = joblib.load(f)
    
# Load Label Encoder
with open('label_encoder.joblib', 'rb') as f:
    le = joblib.load(f)
    
# Load Standard Scaler
with open('standard_scaler.joblib', 'rb') as f:
    sc = joblib.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [request.form[f'feature{i}'] for i in range(1, 9)]
    
    # Label Encoding the categorical columns and standardizing the numerical columns
    for i in range(0, 8):
        if i in [0, 1, 3, 4, 6, 7]:
            features[i] = le.transform([features[i]])[0]
        else:
            features[i] = sc.transform([[features[i]]])[0][0]
    
    prediction = model.predict([features])
    
    return render_template('loan_result.html', prediction=prediction[0])

@app.route('/predict_postman', methods=['POST'])
def predict_postman_function():
    data = request.get_json()
    features = data["data"]
    features = np.array(features).reshape(1, -1)
    
    for i in range(0, 8):
        if i in [0, 1, 3, 4, 6, 7]:
            features[0][i] = le.transform([features[0][i]])[0]
        else:
            features[0][i] = sc.transform([[features[0][i]]])[0][0]
            
    prediction = model.predict(features)
    pred_value = int(prediction[0])
    return jsonify({"prediction": pred_value})

if __name__ == '__main__':
    app.run(debug=True)
    