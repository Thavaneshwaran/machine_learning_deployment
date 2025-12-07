from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# -------------------------
# Load trained artifacts
# -------------------------
loaded_model = joblib.load("model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

FEATURES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    if isinstance(data, dict):
        data = [data]

    df = pd.DataFrame(data)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    df = df[FEATURES]

    scaled = loaded_scaler.transform(df)
    prediction = loaded_model.predict(scaled).tolist()

    return jsonify({"prediction": prediction})


@app.route('/', methods=['GET'])
def home():
    return "🚀 ML Model Deployed Successfully!"


if __name__ == "__main__":
    app.run()
