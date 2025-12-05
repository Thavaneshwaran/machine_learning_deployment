from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# 1. Save the scaler object to a file
joblib.dump(scaler, 'scaler.joblib')
print("Scaler 'scaler.joblib' saved successfully.")

# 2. Load the trained Logistic Regression model
loaded_model = joblib.load('logistic_regression_model.joblib')
print("Model 'logistic_regression_model.joblib' loaded successfully.")

# 3. Load the StandardScaler object
loaded_scaler = joblib.load('scaler.joblib')
print("Scaler 'scaler.joblib' loaded successfully.")

print("Confirmation: Loaded model type - ", type(loaded_model))
print("Confirmation: Loaded scaler type - ", type(loaded_scaler))
# -------------------------
# Load model and scaler ONCE
# -------------------------
try:
    loaded_model = joblib.load("model.pkl")
    loaded_scaler = joblib.load("scaler.pkl")
except Exception as e:
    raise RuntimeError(f"Failed to load model/scaler: {e}")

# Expected feature order (VERY IMPORTANT)
FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


@app.route('/predict', methods=['POST'])
def predict():

    # 1. Validate JSON input
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    # 2. Ensure input is list format
    if isinstance(data, dict):
        data = [data]

    # 3. Convert to DataFrame
    try:
        df = pd.DataFrame(data)
    except Exception:
        return jsonify({'error': 'Invalid input format'}), 400

    # 4. Validate required features
    missing_cols = [col for col in FEATURES if col not in df.columns]
    if missing_cols:
        return jsonify({'error': f'Missing required fields: {missing_cols}'}), 400

    # 5. Reorder columns exactly as training
    df = df[FEATURES]

    # 6. Scale data
    try:
        scaled_input = loaded_scaler.transform(df)
    except Exception as e:
        return jsonify({'error': f'Scaling failed: {e}'}), 500

    # 7. Predict
    try:
        prediction = loaded_model.predict(scaled_input).tolist()
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

    return jsonify({'prediction': prediction})


# -------------------------
# Run Flask App
# -------------------------
if __name__ == '__main__':
    app.run(debug=True)
