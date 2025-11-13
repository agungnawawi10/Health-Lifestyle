from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from utils.rekomendation import get_recomendation

app = Flask(__name__)

# LOAD MODEL dengan error handling yang lebih baik
model = None
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Model loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"❌ Model file not found at {model_path}")
    print("Run 'python train_model.py' to create the model first.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route("/")
def home():
    return jsonify({
        "message": "Healthy Lifestyle Prediction API is running",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "POST /predict - Get disease risk prediction",
            "health": "GET /health - Check API health"
        }
    })

@app.route("/health")
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    # Validasi model
    if model is None:
        return jsonify({
            "error": "Model not loaded", 
            "message": "Please ensure model.pkl exists and run 'python train_model.py' if needed"
        }), 500
    
    # Validasi content type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json()
    
    # Validasi input data
    required_fields = [
        "age", "gender", "bmi", "daily_steps", "sleep_hours", 
        "water_intake_l", "calories_consumed", "smoker", "alcohol", 
        "resting_hr", "systolic_bp", "diastolic_bp", "cholesterol", "family_history"
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({
            "error": "Missing required fields", 
            "missing_fields": missing_fields,
            "required_fields": required_fields
        }), 400
    
    try:
        # Ambil semua data input dari JSON dengan validasi tipe
        age = float(data["age"])
        gender = 1 if str(data.get("gender", "")).lower() == "female" else 0
        bmi = float(data["bmi"])
        daily_steps = float(data["daily_steps"])
        sleep_hours = float(data["sleep_hours"])
        water_intake_l = float(data["water_intake_l"])
        calories = float(data["calories_consumed"])
        smoker = int(data["smoker"])
        alcohol = int(data["alcohol"])
        resting_hr = float(data["resting_hr"])
        systolic_bp = float(data["systolic_bp"])
        diastolic_bp = float(data["diastolic_bp"])
        cholesterol = float(data["cholesterol"])
        family_history = int(data["family_history"])
        
        # Validasi range nilai (opsional tapi disarankan)
        if not (0 <= age <= 150):
            return jsonify({"error": "Age must be between 0-150"}), 400
        if not (10 <= bmi <= 60):
            return jsonify({"error": "BMI must be between 10-60"}), 400
        if smoker not in [0, 1]:
            return jsonify({"error": "Smoker must be 0 or 1"}), 400
        if alcohol not in [0, 1]:
            return jsonify({"error": "Alcohol must be 0 or 1"}), 400
        if family_history not in [0, 1]:
            return jsonify({"error": "Family history must be 0 or 1"}), 400

        # Bentuk array input untuk model
        features = np.array([[
            age, gender, bmi, daily_steps, sleep_hours, water_intake_l,
            calories, smoker, alcohol, resting_hr, systolic_bp,
            diastolic_bp, cholesterol, family_history
        ]], dtype=float)

        # Prediksi dengan probabilitas
        prediction = int(model.predict(features)[0])
        prediction_proba = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else None

        # Dapat Rekomendasi gaya hidup
        recommendation = get_recomendation(prediction)

        # Format respon lengkap
        response = {
            "disease_risk": "High" if prediction == 1 else "Low",
            "prediction_label": prediction,
            "prediction_probability": {
                "low_risk": prediction_proba[0] if prediction_proba else None,
                "high_risk": prediction_proba[1] if prediction_proba else None
            } if prediction_proba else None,
            "recommendation": recommendation,
            "input_features": {
                "age": age, "gender": "Female" if gender == 1 else "Male", 
                "bmi": bmi, "daily_steps": daily_steps, "sleep_hours": sleep_hours,
                "water_intake_l": water_intake_l, "calories_consumed": calories,
                "smoker": smoker, "alcohol": alcohol, "resting_hr": resting_hr,
                "systolic_bp": systolic_bp, "diastolic_bp": diastolic_bp,
                "cholesterol": cholesterol, "family_history": family_history
            }
        }
        return jsonify(response)

    except ValueError as ve:
        return jsonify({"error": f"Invalid data type: {str(ve)}"}), 400
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)