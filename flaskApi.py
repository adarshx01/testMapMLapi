from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS

# Load the trained ML model
with open('prediction_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow cross-origin requests

# Prediction route
@app.route('/predict_safety', methods=['POST'])
def predict_safety():
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

    return jsonify({'SafeRoad': scale_value(safety_score, original_min, original_max, new_min, new_max)})

if __name__ == '__main__':
    app.run(debug=True)
