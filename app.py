# ...existing code...
from flask import Flask, request, jsonify
import pickle
import numpy as np
from utils.rekomendation import get_recomendation

app = Flask(__name__)

# LOAD MODEL
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("Model loaded successfully.")
except Exception:
    model = None
    print("Model file not found. Make sure 'model.pkl' is in the same folder.")


@app.route("/")
def home():
    return jsonify({"message": "Healthy Lifestyle Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        # Ambil semua data input dari JSON
        age = data["age"]
        gender = 1 if str(data.get("gender", "")).lower() == "female" else 0
        bmi = data["bmi"]
        daily_steps = data["daily_steps"]
        sleep_hours = data["sleep_hours"]
        water_intake_l = data["water_intake_l"]
        calories = data["calories_consumed"]
        smoker = data["smoker"]
        alcohol = data["alcohol"]
        resting_hr = data["resting_hr"]
        systolic_bp = data["systolic_bp"]
        diastolic_bp = data["diastolic_bp"]
        cholesterol = data["cholesterol"]
        family_history = data["family_history"]

        # Bentuk array input untuk model (pastikan urutan sesuai yang dipakai saat pelatihan)
        features = np.array(
            [
                [
                    age,
                    gender,
                    bmi,
                    daily_steps,
                    sleep_hours,
                    water_intake_l,
                    calories,
                    smoker,
                    alcohol,
                    resting_hr,
                    systolic_bp,
                    diastolic_bp,
                    cholesterol,
                    family_history,
                ]
            ]
        )

        # Jika belum ada model, kirim pesan error
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Prediksi Risiko penyakit
        prediction = model.predict(features)[0]

        # Dapat Rekomendasi gaya hidup
        recommendation = get_recomendation(prediction)

        # Format respon
        response = {
            "disease_risk": "High" if int(prediction) == 1 else "Low",
            "recommendation": recommendation,
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
