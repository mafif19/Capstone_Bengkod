import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("best_random_forest_model.pkl")

# Rata-rata dan standar deviasi dari data training (untuk standardisasi)
means = {'Age': 23.144453, 'Height': 1.715131, 'Weight': 92.504880, 'FCVC': 2.456863,
         'CH2O': 2.060705, 'FAF': 1.031469, 'TUE': 0.693352}
stds = {'Age': 4.102400, 'Height': 0.087400, 'Weight': 27.473986, 'FCVC': 0.546924,
        'CH2O': 0.601654, 'FAF': 0.855703, 'TUE': 0.586877}

def standardize(value, mean, std):
    return (value - mean) / std

def tampilkan_hasil_prediksi(label_prediksi):
    info = {
        "Insufficient_Weight": {
            "desc": "Anda termasuk dalam kategori Berat Badan Kurang. Ini berarti tubuh Anda mungkin memerlukan asupan gizi yang lebih untuk mencapai berat badan sehat.",
            "rekomendasi": "Disarankan untuk berkonsultasi dengan ahli gizi untuk pola makan yang sesuai dan menjaga kesehatan secara keseluruhan.",
            "color": "blue"
        },
        "Normal_Weight": {
            "desc": "Berat badan Anda berada pada kisaran normal yang sehat. Pertahankan pola hidup aktif dan konsumsi makanan bergizi seimbang.",
            "rekomendasi": "Lanjutkan gaya hidup sehat dan rutin cek kesehatan secara berkala.",
            "color": "green"
        },
        "Overweight_Level_I": {
            "desc": "Anda masuk dalam kategori Kelebihan Berat Badan Tingkat I. Ini adalah peringatan awal untuk mulai memperhatikan pola makan dan aktivitas fisik.",
            "rekomendasi": "Disarankan untuk meningkatkan aktivitas fisik dan mengurangi konsumsi makanan tinggi kalori secara bertahap.",
            "color": "yellow"
        },
        "Overweight_Level_II": {
            "desc": "Kategori Kelebihan Berat Badan Tingkat II. Risiko masalah kesehatan mulai meningkat jika tidak ada perubahan gaya hidup.",
            "rekomendasi": "Segera konsultasikan dengan profesional kesehatan dan buatlah rencana diet serta olahraga yang terstruktur.",
            "color": "orange"
        },
        "Obesity_Type_I": {
            "desc": "Anda termasuk Obesitas Tipe I, yang berarti ada penumpukan lemak berlebih yang dapat meningkatkan risiko penyakit kronis.",
            "rekomendasi": "Konsultasi dengan dokter atau ahli gizi sangat dianjurkan untuk memulai program penurunan berat badan yang aman dan efektif.",
            "color": "orange"
        },
        "Obesity_Type_II": {
            "desc": "Obesitas Tipe II, kondisi ini sudah termasuk tingkat berat dengan risiko kesehatan yang signifikan.",
            "rekomendasi": "Perubahan gaya hidup dan pengawasan medis yang ketat diperlukan untuk menghindari komplikasi serius.",
            "color": "red"
        },
        "Obesity_Type_III": {
            "desc": "Obesitas Tipe III (Obesitas Morbid) sangat serius dan memerlukan intervensi medis segera.",
            "rekomendasi": "Segera lakukan konsultasi dengan dokter spesialis untuk penanganan yang tepat, bisa meliputi terapi medis atau operasi jika diperlukan.",
            "color": "red"
        }
    }

    hasil = info.get(label_prediksi, None)
    if hasil:
        st.markdown(f"<h3 style='color:{hasil['color']};'>Hasil Prediksi: {label_prediksi.replace('_', ' ')}</h3>", unsafe_allow_html=True)
        st.write(hasil['desc'])
        st.markdown(f"**Rekomendasi:** {hasil['rekomendasi']}")
    else:
        st.error("Terjadi kesalahan dalam prediksi. Silakan coba lagi.")

label_map = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

st.title("Prediksi Kategori Obesitas")

# Input
age = st.number_input("Usia (tahun)", min_value=10, max_value=100)
if age > 35 or age < 14:
    st.warning("Model ini dilatih pada data usia 14 hingga 35 tahun. Hasil prediksi untuk usia di luar rentang ini mungkin tidak akurat.")

gender = st.radio("Jenis Kelamin", [0, 1], format_func=lambda x: "Perempuan" if x == 0 else "Laki-laki")

height = st.number_input("Tinggi Badan (meter)", min_value=1.20, max_value=2.20, step=0.01)
if height < 1.45 or height > 1.98:
    st.info("Model dilatih dengan tinggi badan antara 1.45 m hingga 1.98 m. Prediksi di luar rentang ini mungkin kurang tepat.")

weight = st.number_input("Berat Badan (kg)", min_value=30.0, max_value=200.0)
if weight < 39.0 or weight > 173.0:
    st.info("Data pelatihan mencakup berat antara 39 kg hingga 173 kg. Di luar rentang ini, prediksi bisa kurang akurat.")

favc = st.selectbox("Sering Konsumsi Makanan Tinggi Kalori?", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

fcvc = st.number_input("Frekuensi Konsumsi Sayur (1 - 3)", min_value=1.0, max_value=3.0, step=0.1)

calc = st.selectbox("Konsumsi Alkohol", [0, 1, 2], format_func=lambda x: ["Tidak Pernah", "Kadang", "Sering"][x])

scc = st.selectbox("Konsultasi Gizi", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

ch2o = st.number_input("Konsumsi Air Harian (1 - 3 liter)", min_value=1.0, max_value=4.0, step=0.1)
if ch2o < 1.0 or ch2o > 3.0:
    st.info("Model dilatih dengan konsumsi air antara 1 hingga 3 liter. Nilai di luar rentang ini dapat memengaruhi akurasi prediksi.")

history = st.selectbox("Riwayat Keluarga Obesitas", [0, 1], format_func=lambda x: "Tidak" if x == 0 else "Ya")

faf = st.number_input("Aktivitas Fisik Mingguan (jam)", min_value=0.0, max_value=6.0, step=0.1)
if faf < 0.0 or faf > 3.0:
    st.info("Aktivitas fisik dalam data pelatihan berkisar 0 hingga 3 jam. Di atas itu, akurasi model bisa berkurang.")

tue = st.number_input("Waktu di Depan Layar per Hari (jam)", min_value=0.0, max_value=10.0, step=0.1)
if tue < 0.0 or tue > 2.0:
    st.info("Model dilatih dengan waktu layar harian antara 0 hingga 2 jam. Nilai lebih dari itu dapat menghasilkan prediksi yang kurang akurat.")

caec = st.selectbox("Frekuensi Makan Berlebihan", [0, 1, 2, 3], format_func=lambda x: ["Tidak Pernah", "Kadang", "Sering", "Selalu"][x])

mtrans = st.selectbox("Transportasi Harian", [0, 1, 2, 3, 4], format_func=lambda x: ["Transportasi Umum", "Jalan Kaki", "Mobil", "Motor", "Sepeda"][x])

if st.button("Prediksi"):
    # Siapkan input terstandardisasi
    X_input = [
        standardize(age, means['Age'], stds['Age']),
        gender,
        standardize(height, means['Height'], stds['Height']),
        standardize(weight, means['Weight'], stds['Weight']),
        favc,
        standardize(fcvc, means['FCVC'], stds['FCVC']),
        calc,
        scc,
        standardize(ch2o, means['CH2O'], stds['CH2O']),
        history,
        standardize(faf, means['FAF'], stds['FAF']),
        standardize(tue, means['TUE'], stds['TUE']),
        caec,
        mtrans
    ]

    pred = model.predict([X_input])[0]
    label_prediksi = label_map[pred]

    tampilkan_hasil_prediksi(label_prediksi)
