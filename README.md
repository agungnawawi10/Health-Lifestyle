// ...existing code...
# Health-Lifestyle

Ringkasan singkat
-----------------
Health-Lifestyle adalah proyek deteksi risiko penyakit berdasarkan pola hidup (usia, BMI, aktivitas, tidur, konsumsi, tekanan darah, kolesterol, dll). Model dilatih dengan RandomForest dan tersedia sebuah API Flask untuk inferensi.

Fitur utama
-----------
- Prediksi risiko penyakit (High / Low)
- API RESTful: endpoint `/predict` (POST JSON)
- Skrip pelatihan (`train_model.py`) dan diagnostik (`diagnose_model.py`)
- Rekomendasi gaya hidup otomatis dari `utils/rekomendation.py`

Quick start (Windows)
---------------------
1. Buat virtual environment dan install dependensi:
   - python -m venv .venv
   - .venv\Scripts\activate
   - pip install -r requirements.txt

2. Latih model (simpan `model.pkl`):
   - python train_model.py

3. Jalankan server:
   - python app.py
   - API tersedia di http://localhost:5000

Contoh request ke API
---------------------
POST /predict dengan Content-Type: application/json

Contoh body:
{
  "age": 70,
  "gender": "Male",
  "bmi": 38.5,
  "daily_steps": 1500,
  "sleep_hours": 4.0,
  "water_intake_l": 1.0,
  "calories_consumed": 3800,
  "smoker": 1,
  "alcohol": 1,
  "resting_hr": 95,
  "systolic_bp": 170,
  "diastolic_bp": 110,
  "cholesterol": 320,
  "family_history": 1
}

Diagnosa & catatan penting
--------------------------
- Dataset berisi ketidakseimbangan kelas (sekitar 75% label 0). Ini membuat model cenderung menebak kelas mayoritas.
- Rekomendasi perbaikan: stratified split, class_weight='balanced' atau oversampling (SMOTE), feature engineering, evaluasi dengan F1/AUC, dan tuning threshold prediksi.
- Gunakan `diagnose_model.py` untuk melihat distribusi label, confusion matrix, dan probabilitas prediksi per-sampel.

Struktur project
----------------
- app.py — API Flask
- train_model.py — skrip pelatihan
- diagnose_model.py — skrip diagnosa (opsional)
- data/health_lifestyle_dataset.csv — data latih
- utils/rekomendation.py — fungsi rekomendasi
- model.pkl — model terlatih (di-generate oleh train_model.py)

Kontribusi
----------
1. Fork repo
2. Buat branch fitur (`feature/...`)
3. Test dan ajukan pull request

Lisensi
-------
MIT — gunakan dan kembangkan sesuai kebutuhan (cantumkan atribusi jika perlu).