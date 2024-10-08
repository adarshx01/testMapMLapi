import os
from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import logging

print("Starting app initialization...")

# Load the trained ML model
model_path = 'prediction_model.pkl'
print(f"Attempting to load model from {model_path}")
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully")
else:
    print(f"Error: {model_path} not found")
    model = None

app = Flask(__name__)

# CORS configuration
CORS(app)

# Logging configuration
app.logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)

@app.route('/')
def home():
    return "Hello, World!"

# Prediction route
@app.route('/predict_safety', methods=['POST'])
def predict_safety():
    app.logger.debug("Received prediction request")
    try:
        data = request.json
        features = np.array([
            data['Magnitude'],
            data['Crime_Types'],
            data['time_of_day'],
            data['shops_nearby'],
            data['area_type'],
            data['has_Vehicle'],
            data['crime_rate'],
            data['number_crime_last_Three_months'],
            data['number_people_accompanying'],
            data['weather_condition'],
            data['proximity_police_station'],
            data['proximity_hospital'],
            data['streetlight'],
            data['traffic_density'],
            data['reported_crimes'],
            data['proximity_public_transport']
        ]).reshape(1, -1)

        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        safety_score = model.predict(features)[0]

        def scale_value(original_value, original_min, original_max, new_min, new_max):
            original_range = original_max - original_min
            new_range = new_max - new_min
            scaled_value = ((original_value - original_min) / original_range) * new_range + new_min
            return scaled_value

        original_min = 34.41
        original_max = 56.41
        new_min = 0
        new_max = 100

        scaled_score = scale_value(safety_score, original_min, original_max, new_min, new_max)
        app.logger.info(f"Prediction made: {scaled_score}")
        return jsonify({'SafeRoad': scaled_score})
    except Exception as e:
        app.logger.error(f"Error in predict_safety: {str(e)}")
        return jsonify({"error": "An error occurred during prediction"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": model is not None}), 200

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({"error": "An unexpected error occurred"}), 500

print("App initialization complete")

if __name__ == '__main__':
    app.run(debug=True)