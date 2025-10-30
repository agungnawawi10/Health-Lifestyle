def get_recomendation(prediction):
    if prediction == 1:
        return (
            "Risiko penyakit kamu tergolong tinggi. "
            "Perhatikan pola makan, rutin berolahraga minimal 30 menit per hari, "
            "kurangi konsumsi rokok dan alkohol, serta cukup tidur (7–8 jam) yaa."
        )
    else:
        return (
            "Risiko penyakit kamu tergolong rendah. "
            "Pertahankan gaya hidup sehat, cukup minum air, "
            "dan tetap aktif bergerak setiap hari!"
                        
        )
