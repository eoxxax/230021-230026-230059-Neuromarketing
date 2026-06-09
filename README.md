# USER GUIDE

## NeuroMarketing EEG Analyzer

**Disusun oleh:**
| Nama | NIM |
|---|---|
| Senia Nur Hasanah | 140810230021 |
| Audrey Shaina Tjandra | 140810230026 |
| Siti Nailah Eko Putri Alawiyah | 140810230059 |

Program Studi S-1 Teknik Informatika — FMIPA Universitas Padjadjaran

---

### 1. Pendahuluan

NeuroMarketing EEG Analyzer merupakan aplikasi berbasis Streamlit yang digunakan untuk menganalisis tingkat perhatian (attention) responden berdasarkan data Electroencephalography (EEG). Sistem memanfaatkan model hybrid CNN-LSTM dengan Fuzzy Layer (ANFIS-inspired) untuk mengidentifikasi kondisi mata terbuka (eye-open) dan mata tertutup (eye-closed), kemudian mengubah hasil prediksi menjadi Attention Score yang dapat digunakan dalam analisis neuromarketing.

---

## 2. Persyaratan Sistem

### Perangkat Lunak

* Python 3.10 atau lebih baru
* Streamlit
* TensorFlow
* Scikit-Learn
* Pandas
* NumPy
* Plotly

### File yang Dibutuhkan

Pastikan file berikut tersedia dalam folder aplikasi:

* `app.py`
* `model_eeg.h5`
* `scaler_eeg.pkl`

---

## 3. Menjalankan Aplikasi

Buka terminal pada folder proyek kemudian jalankan perintah:

```bash
streamlit run app.py
```

Setelah berhasil dijalankan, aplikasi akan terbuka melalui browser.

---

## 4. Struktur Data EEG

Aplikasi menerima file berformat CSV dengan 14 kanal EEG berikut:

| No | Channel EEG |
| -- | ----------- |
| 1  | AF3         |
| 2  | AF4         |
| 3  | F3          |
| 4  | F4          |
| 5  | F7          |
| 6  | F8          |
| 7  | FC5         |
| 8  | FC6         |
| 9  | T7          |
| 10 | T8          |
| 11 | P7          |
| 12 | P8          |
| 13 | O1          |
| 14 | O2          |

Catatan:

* Kolom label dapat disertakan atau dikosongkan.
* Format file harus berupa `.csv`.

---

## 5. Pengaturan Analisis

Pada panel sebelah kiri (sidebar), pengguna dapat mengatur parameter analisis berikut.

### a. Sampling Rate (Hz)

Menentukan frekuensi pengambilan data EEG.

Pilihan:

* 128 Hz
* 256 Hz
* 512 Hz

Gunakan nilai yang sesuai dengan perangkat EEG yang digunakan.

### b. Durasi Segmen

Menentukan panjang jendela waktu untuk perhitungan Attention Score.

Rentang:

* 2–30 detik

Rekomendasi:

* 5 detik untuk analisis normal.
* 10–30 detik untuk analisis yang lebih stabil.

### c. Threshold Prediksi

Menentukan batas klasifikasi model.

Rentang:

* 0.3 – 0.7

Interpretasi:

* Probabilitas di bawah threshold → Mata terbuka (attention tinggi)
* Probabilitas di atas threshold → Mata tertutup (attention rendah)

---

## 6. Langkah Penggunaan

### Langkah 1 – Upload Data EEG

1. Buka tab **Upload & Prediksi**.
2. Klik tombol **Browse Files**.
3. Pilih file EEG berformat CSV.
4. Pastikan seluruh kanal EEG tersedia.

### Langkah 2 – Jalankan Analisis

1. Klik tombol **Mulai Analisis EEG**.
2. Sistem akan:

   * Melakukan preprocessing data.
   * Melakukan normalisasi menggunakan scaler.
   * Membentuk sequence EEG.
   * Menjalankan prediksi menggunakan model hybrid CNN-LSTM dengan Fuzzy Layer (ANFIS-inspired).

### Langkah 3 – Melihat Hasil

Setelah proses selesai, sistem akan menampilkan:

* Attention Score
* Eye Open Ratio
* Kategori perhatian
* Grafik attention
* Statistik analisis

---

## 7. Dashboard Attention

Pada tab **Dashboard Attention**, pengguna dapat melihat:

### Attention Score

Menunjukkan tingkat perhatian responden dalam bentuk persentase.

Kategori:

| Attention Score | Kategori |
| --------------- | -------- |
| ≥ 60%           | Tinggi   |
| 40% – 59%       | Sedang   |
| < 40%           | Rendah   |

Attention Score dihitung berdasarkan proporsi prediksi kondisi Eye Open pada setiap segmen EEG yang dianalisis. Semakin tinggi proporsi Eye Open yang terdeteksi, semakin tinggi nilai Attention Score yang dihasilkan sistem.

### Eye Open Ratio

Menunjukkan persentase waktu mata responden dalam kondisi terbuka.

Semakin tinggi nilainya, semakin tinggi tingkat perhatian responden.

### Grafik Attention

Menampilkan perubahan tingkat perhatian dari waktu ke waktu berdasarkan segmentasi EEG.

---

## 8. Performa Model

Pada tab **Performa Model**, pengguna dapat melihat evaluasi model yang digunakan.

Informasi yang ditampilkan meliputi:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Evaluasi ini digunakan untuk mengetahui kualitas model dalam mengklasifikasikan kondisi mata terbuka dan mata tertutup.

---

## 9. Tentang Model

Aplikasi menggunakan arsitektur hybrid:

### CNN (Convolutional Neural Network)

Digunakan untuk mengekstraksi pola spasial dari sinyal EEG.

### LSTM (Long Short-Term Memory)

Digunakan untuk menangkap hubungan temporal antar sinyal EEG.

### ANFIS (Adaptive Neuro Fuzzy Inference System)

Digunakan untuk meningkatkan kemampuan interpretasi dan pengambilan keputusan berbasis logika fuzzy.

Kombinasi CNN, LSTM, dan Fuzzy Layer menghasilkan model hybrid yang mampu mengidentifikasi kondisi Eye Open dan Eye Closed dari sinyal EEG. Hasil prediksi kemudian digunakan untuk menghitung Attention Score sebagai indikator tingkat perhatian responden.

---

## 10. Troubleshooting

### Model Tidak Ditemukan

Penyebab:

* File model tidak berada pada folder aplikasi.

Solusi:

* Pastikan file `model_eeg.h5` dan `scaler_eeg.pkl` tersedia dalam direktori yang sama dengan `app.py`.

### Format CSV Tidak Sesuai

Penyebab:

* Nama kolom berbeda dari format yang dipersyaratkan.

Solusi:

* Sesuaikan nama kolom dengan 14 channel EEG yang digunakan sistem.

### Analisis Gagal

Penyebab:

* Data kosong atau jumlah sampel terlalu sedikit.

Solusi:

* Pastikan data EEG memiliki jumlah sampel yang memadai dan tidak mengandung banyak nilai kosong.

---

## 11. Kesimpulan

NeuroMarketing EEG Analyzer membantu peneliti dan praktisi neuromarketing dalam mengevaluasi tingkat perhatian responden menggunakan sinyal EEG secara otomatis. Dengan memanfaatkan model hybrid CNN-LSTM dengan Fuzzy Layer (ANFIS-inspired), sistem mampu menghasilkan Attention Score, visualisasi perhatian, serta informasi pendukung yang dapat digunakan untuk analisis efektivitas stimulus pemasaran.

---

## Mata Kuliah

**Soft Computing** — UTS Genap 2025/2026  
Dosen: Dr. Ir. Intan Nurma Yulita, M.T  
Fakultas MIPA — Teknik Informatika
Universitas Padjadjaran

