from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import IsolationForest
import numpy as np

from simulate_data import simulate_health_data, inject_anomalies

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load and train the model on startup
print("Training model...")
# Generate a broader range of data for more robust training
training_data = simulate_health_data(1000)
# Add some anomalous data to the training set to help the model learn
training_data = inject_anomalies(training_data, num=50)
features = training_data[['heart_rate', 'blood_oxygen']]
# Adjust contamination to reflect the approximate proportion of anomalies
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(features)
print("Model training complete.")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict anomalies in real-time.
    """
    data = request.get_json()

    # Input validation
    if not data or 'heart_rate' not in data or 'blood_oxygen' not in data:
        return jsonify({"error": "Missing 'heart_rate' or 'blood_oxygen' in request"}), 400

    try:
        heart_rate = float(data['heart_rate'])
        blood_oxygen = float(data['blood_oxygen'])
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input type. 'heart_rate' and 'blood_oxygen' must be numbers."}), 400

    # Prepare data for prediction
    input_data = np.array([[heart_rate, blood_oxygen]])

    # Make prediction
    prediction = model.predict(input_data)
    status = 'Anomaly' if prediction[0] == -1 else 'Normal'

    return jsonify({"status": status})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
