from flask import Flask, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)
# Load the trained model and scaler
with open("heart_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)
with open("heart_scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Expecting JSON input
        features = np.array(data["features"]).reshape(1, -1)  # Convert to 2D array
        features = scaler.transform(features)  # Scale input
        prediction = model.predict(features)[0]  # Get prediction
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})
if __name__ == "__main__":
    app.run(debug=True)