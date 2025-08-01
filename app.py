import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template

# --- Inisialisasi Aplikasi ---
app = Flask(__name__)

# --- Memuat Model dan Kolom ---
# Langkah ini penting untuk memastikan model siap saat server berjalan.
try:
    model = joblib.load('model_harga_mobil_final/car_price_model_final.joblib')
    model_columns = joblib.load('model_harga_mobil_final/model_columns_final.joblib')
    print("✅ Model dan kolom berhasil dimuat.")
    # Mencetak beberapa contoh nama kolom untuk debugging awal
    print("Contoh nama kolom dari model:", model_columns[:5])
except Exception as e:
    print(f"❌ FATAL: Gagal memuat model atau file kolom: {e}")
    model = None
    model_columns = []

# --- Halaman Utama (Frontend) ---
@app.route('/')
def home():
    # Flask akan mencari file ini di dalam folder 'templates'
    return render_template('index.html')

# --- Endpoint untuk Prediksi ---
@app.route('/predict', methods=['POST'])
def predict():
    # Pemeriksaan keamanan jika model gagal dimuat
    if not model or not model_columns:
        return jsonify({'error': 'Model tidak siap. Periksa log server.'}), 500

    try:
        # 1. Ambil data JSON dari permintaan web
        data = request.get_json()
        print(f"Data mentah diterima dari web: {data}")

        # 2. Siapkan kerangka data (dictionary) yang bersih
        # Semua fitur diawali dengan nilai 0, sesuai yang diharapkan model.
        feature_dict = {col: 0 for col in model_columns}

        # 3. Isi nilai-nilai numerik
        feature_dict['Year'] = float(data.get('year', 0))
        # Nama kolom 'KM's driven' harus sama persis dengan yang digunakan saat training
        feature_dict["KM's driven"] = float(data.get('mileage', 0))

        # 4. Proses dan cocokkan nilai-nilai kategorikal
        # Ini adalah bagian paling rawan kesalahan.
        
        # Ambil nilai dari form, hapus spasi, dan jadikan huruf kecil
        # Contoh: "toyota", "avanza", "petrol", "manual"
        brand = data.get('brand', '').strip().lower()
        model_name = data.get('model', '').strip().lower()
        fuel = data.get('fuel', '').strip().lower()
        transmission = data.get('transmission', '').strip().lower()

        # Buat nama kolom yang mungkin dari input
        # Contoh: "make_toyota", "model_avanza", dll.
        possible_brand_col = f'make_{brand}'
        possible_model_col = f'model_{model_name}'
        possible_fuel_col = f'fuel_{fuel}'
        possible_transmission_col = f'transmission_{transmission}'

        # Lakukan perbandingan dengan cara yang tidak sensitif huruf besar/kecil
        for col in model_columns:
            col_lower = col.lower()
            if brand and col_lower == possible_brand_col:
                feature_dict[col] = 1
            if model_name and col_lower == possible_model_col:
                feature_dict[col] = 1
            if fuel and col_lower == possible_fuel_col:
                feature_dict[col] = 1
            if transmission and col_lower == possible_transmission_col:
                feature_dict[col] = 1

        # 5. Buat DataFrame final dari dictionary
        final_df = pd.DataFrame([feature_dict])
        
        # Pastikan urutan kolom 100% sama dengan saat model dilatih
        final_df = final_df[model_columns]

        # Debugging: Cetak fitur yang tidak nol sebelum prediksi
        print("\n--- Fitur yang dikirim ke model (nilai != 0) ---")
        print(final_df.loc[:, (final_df != 0).any(axis=0)])
        print("--------------------------------------------------\n")
        
        # 6. Lakukan prediksi
        prediction = model.predict(final_df)
        output = prediction[0]
        
        # 7. Kirim hasil kembali ke web
        return jsonify({'prediction_text': f'Rp {output:,.0f}', 'details': data})

    except Exception as e:
        # Tangani error jika terjadi saat proses prediksi
        print(f"❌ Terjadi error saat prediksi: {e}")
        return jsonify({'error': f'Error pada server: {e}'}), 500

# --- Menjalankan Server ---
if __name__ == "__main__":
    # debug=True akan otomatis me-restart server jika ada perubahan pada file ini
    app.run(debug=True)