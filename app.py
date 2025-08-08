# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
MODEL_PATH = 'model.joblib'
model = joblib.load(MODEL_PATH)

FEATURES = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']

@app.route('/')
def home():
    return render_template('index.html')

def parse_inputs_from_json_or_list(data):
    # data can be {"inputs":[...]} or a list
    if isinstance(data, dict) and 'inputs' in data:
        arr = data['inputs']
    elif isinstance(data, dict):
        # maybe directly keys
        arr = [float(data.get(f, 0)) for f in FEATURES]
    else:
        arr = data
    arr = [float(x) for x in arr]
    return np.array(arr).reshape(1, -1)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
            X = parse_inputs_from_json_or_list(data)
        else:
            # form inputs
            X = np.array([float(request.form.get(f, 0)) for f in FEATURES]).reshape(1, -1)

        prob = float(model.predict_proba(X)[0][1])
        pred = int(model.predict(X)[0])
        return jsonify({'prediction': pred, 'diabetes_probability': prob})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
