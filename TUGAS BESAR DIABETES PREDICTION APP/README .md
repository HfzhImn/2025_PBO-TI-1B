# SICERDIK: Sistem Cegah Risiko Diabetes berbasis Indikator Kesehatan 🩺

---

## 📑 Deskripsi Singkat

**SICERDIK** adalah aplikasi web edukatif berbasis machine learning untuk **prediksi risiko diabetes**.  
Aplikasi ini dikembangkan menggunakan Python, OOP, dan Streamlit, dengan memanfaatkan data survei kesehatan publik BRFSS 2015 untuk memprediksi risiko **tidak diabetes, prediabetes, atau diabetes** berdasarkan indikator kesehatan dan gaya hidup pengguna.

---

## 📋 Fitur Utama

- **Prediksi risiko diabetes 3 kelas:**
  - Tidak Diabetes
  - Prediabetes
  - Diabetes
- **Input data kesehatan interaktif:**  
  Slider, dropdown, radio button, dan penjelasan edukatif.
- **Visualisasi data dan model:**  
  Pie chart distribusi, histogram BMI, feature importance, confusion matrix, dll.
- **Saran otomatis & edukasi:**  
  Saran kesehatan langsung muncul sesuai hasil prediksi.
- **Simulasi "Jika semua orang seperti Anda":**  
  Menampilkan proporsi prediksi pada populasi jika seluruh dataset berisi orang dengan profil yang sama seperti user.
- **Riwayat prediksi & Download:**  
  Menyimpan history prediksi selama sesi dan dapat diunduh sebagai CSV.
- **Halaman penjelasan model & dataset:**  
  Penjelasan kelebihan Random Forest, distribusi label, fitur dataset, dll.
- **UI/UX modern, responsif, nuansa putih-oranye (khas Streamlit).**
- **Pemrograman modular dan OOP:**  
  Menggunakan Class Pasien & ModelHandler agar mudah dikembangkan dan dirawat.

---

## 🗂️ Struktur Direktori Project

diabetes-prediction-app/
│
├─ data/
│ └─ diabetes_012_health_indicators_BRFSS2015.csv
├─ models/
│ ├─ random_forest_model.pkl
│ └─ scaler.pkl
├─ src/
│ ├─ init.py
│ ├─ preprocessing.py
│ ├─ data_mapping.py
│ ├─ pasien.py
│ ├─ model_handler.py
| ├─ train_model.py
│ └─ visualizer.py
|
├─ streamlit_app.py
├─ requirements.txt
└─ README.md

---

## 🔗 Dataset

- **Sumber:** [Kaggle - Diabetes Health Indicators Dataset (BRFSS 2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- **Lisensi:** Public domain
- **Fitur:**  
  21 fitur kesehatan & gaya hidup, target `Diabetes_012` (0 = Tidak Diabetes, 1 = Prediabetes, 2 = Diabetes)

---

## 🚀 Cara Menjalankan Aplikasi

### 1. **Kloning repositori (atau download project)**

```bash
git clone https://github.com/HfzhImn/2025_PBO-TI-1B/diabetes-prediction-app.git
cd diabetes-prediction-app
```

### 2. **Aktifkan virtual environment & install dependencies**

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
pip install streamlit pandas scikit-learn imbalanced-learn matplotlib seaborn
```

### 3.**Unduh dataset dari Kaggle, letakkan di folder /data**

File: diabetes_012_health_indicators_BRFSS2015.csv

### 4. **Lakukan training model (sekali di awal atau setelah ganti kode/data)**

```bash
python src/train_model.py
```

### 5. **Jalankan aplikasi Streamlit**

```bash
streamlit run streamlit_app.py
```

Aplikasi akan dapat diakses pada browser di alamat http://localhost:8501

---

## 🧩 Penjelasan Fitur dan Alur Aplikasi

A. Input Data User  
User mengisi data kesehatan melalui form sidebar (menggunakan tombol submit, tidak auto-refresh).  
Data input user diproses sebagai objek Pasien (OOP).

B. Proses Prediksi  
Data user dinormalisasi sesuai dengan training.  
Model Random Forest (yang telah di-balance dengan SMOTE+Undersampling) melakukan prediksi kelas dan probabilitas.  
Hasil, probabilitas, dan saran edukasi langsung ditampilkan.

C. Visualisasi  
Pie chart distribusi label.  
Histogram BMI (dengan garis marker posisi user).  
Feature importance (barplot).  
Korelasi fitur dengan target.  
Confusion matrix (warna oranye).

D. Simulasi  
Jika semua data pada dataset diisi dengan profil user, model akan menampilkan proporsi prediksi pada populasi dataset.

E. History & Download  
Seluruh prediksi selama sesi disimpan, dan dapat diunduh sebagai file CSV.

F. Halaman Edukasi  
Penjelasan tentang dataset, fitur, dan model ML yang digunakan.  
Artikel edukasi singkat seputar diabetes.

---

## ⚙️ Modularisasi & OOP

src/pasien.py:  
Class Pasien — untuk input user, dapat dikembangkan untuk history, validasi, dll.

src/model_handler.py:  
Class ModelHandler — membungkus seluruh workflow ML (training, balancing data, evaluasi, prediksi, simpan/load model).

src/visualizer.py:  
Fungsi visualisasi (pie chart, barplot, histogram, confusion matrix).

src/preprocessing.py:  
Pembersihan dan split data.

src/data_mapping.py:  
Mapping deskripsi fitur kategori menjadi label yang lebih mudah dipahami user.

---

### 🔬 Penjelasan Model Machine Learning

Model: Random Forest Classifier (n_estimators=100, class_weight='balanced')  
Balancing: Kombinasi SMOTE + RandomUndersampling pada training set  
Scaling: StandardScaler (fitur numerik dinormalisasi)  
Evaluasi: Classification report (precision, recall, f1-score per kelas) & confusion matrix  
Target: Tiga kelas (0 = Tidak Diabetes, 1 = Prediabetes, 2 = Diabetes)

---

### 👨‍💻 Konsep OOP dalam Aplikasi

Input user diproses dalam bentuk objek (Class Pasien).  
Workflow training, evaluasi, dan prediksi model dibungkus dalam ModelHandler, sehingga lebih modular dan mudah dikembangkan di masa depan.

---

### 📚 Referensi

Kaggle: Diabetes Health Indicators Dataset  
Streamlit Docs  
Imbalanced-learn: SMOTE & Undersampling  
Scikit-learn RandomForestClassifier

---

### ❗ Disclaimer

Aplikasi ini bertujuan untuk edukasi dan deteksi awal.  
Tidak menggantikan diagnosis medis.  
Selalu konsultasikan hasil prediksi dengan tenaga kesehatan profesional.

---

### 🙋 Pengembang & Kontributor

Hafizh Iman Wicaksono (2025)  
IG: @hafizhiman  
Tugas Besar Mata Kuliah Pemrograman Berbasis Objek — Politeknik Negeri Semaranng, Prodi D4-Teknologi Rekayasa Komputer
