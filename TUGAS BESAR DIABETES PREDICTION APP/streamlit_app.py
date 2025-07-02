import streamlit as st
import pandas as pd
from src.data_mapping import age_mapping, education_mapping, income_mapping, target_mapping
from src.pasien import Pasien
from src.model_handler import ModelHandler
from src.visualizer import (
    plot_target_distribution, plot_feature_importance, plot_bmi_histogram_with_user, plot_corr_with_target, plot_confusion_matrix
)
import matplotlib.pyplot as plt

# ------------------- STYLING: Warna Putih & Orange ---------------------
st.set_page_config(page_title="SICERDIK", page_icon="🩺", layout="wide")

# Custom CSS warna pada Streamlit
st.markdown("""
    <style>
    body, .main { background-color: #ffffff; color: #FF8303; }
    .stButton>button { background: #FF8303; color: #fff; border-radius: 10px; font-weight: bold; }
    div[data-testid="stSidebar"] { background: #fff7ef; }
    h1, h2, h3, h4, h5, h6 { color: #FFFFFF !important; }
    </style>
""", unsafe_allow_html=True)

# ------------------- SESSION STATE HISTORY ---------------------------
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ------------------- FUNGSI INPUT -----------------------
def user_input_features_form():
    with st.sidebar.form(key='form_prediksi'):
        st.markdown("### 📝 **Silahkan Input Data Kesehatan Anda**")
        data = {}
        data['HighBP'] = st.radio("Tekanan Darah Tinggi?", [1,0], format_func=lambda x: "✅ Ya" if x==1 else "❌ Tidak")
        data['HighChol'] = st.radio("Kolesterol Tinggi?", [1,0], format_func=lambda x: "✅ Ya" if x==1 else "❌ Tidak")
        data['CholCheck'] = st.radio("Pernah Cek Kolesterol?", [1,0], format_func=lambda x: "✅ Ya" if x==1 else "❌ Tidak")
        data['BMI'] = st.slider("BMI (Body Mass Index)", 12, 60, 25)
        data['Smoker'] = st.radio("Pernah Merokok?", [1,0], format_func=lambda x: "🚬 Ya" if x==1 else "❌ Tidak")
        data['Stroke'] = st.radio("Pernah Stroke?", [1,0], format_func=lambda x: "⚡ Ya" if x==1 else "❌ Tidak")
        data['HeartDiseaseorAttack'] = st.radio("Riwayat Jantung?", [1,0], format_func=lambda x: "❤️ Ya" if x==1 else "❌ Tidak")
        data['PhysActivity'] = st.radio("Aktif Olahraga?", [1,0], format_func=lambda x: "🏃‍♂️ Ya" if x==1 else "❌ Tidak")
        data['Fruits'] = st.radio("Konsumsi Buah Harian?", [1,0], format_func=lambda x: "🍎 Ya" if x==1 else "❌ Tidak")
        data['Veggies'] = st.radio("Konsumsi Sayur Harian?", [1,0], format_func=lambda x: "🥦 Ya" if x==1 else "❌ Tidak")
        data['HvyAlcoholConsump'] = st.radio("Konsumsi Alkohol Berat?", [1,0], format_func=lambda x: "🍻 Ya" if x==1 else "❌ Tidak")
        data['AnyHealthcare'] = st.radio("Akses Kesehatan?", [1,0], format_func=lambda x: "🏥 Ya" if x==1 else "❌ Tidak")
        data['NoDocbcCost'] = st.radio("Pernah tidak ke dokter karena biaya?", [1,0], format_func=lambda x: "💸 Ya" if x==1 else "❌ Tidak")
        data['GenHlth'] = st.slider("Penilaian Kesehatan Umum (1=Baik, 5=Buruk)", 1, 5, 3)
        data['MentHlth'] = st.slider("Jumlah Hari Mental Kurang Sehat (30 hari)", 0, 30, 0)
        data['PhysHlth'] = st.slider("Jumlah Hari Fisik Kurang Sehat (30 hari)", 0, 30, 0)
        data['DiffWalk'] = st.radio("Sulit Jalan?", [1,0], format_func=lambda x: "🚶‍♂️ Ya" if x==1 else "❌ Tidak")
        data['Sex'] = st.radio("Jenis Kelamin", [0,1], format_func=lambda x: "👨 Pria" if x==0 else "👩 Wanita")
        data['Age'] = st.selectbox("Usia", options=list(age_mapping.keys()), format_func=lambda x: age_mapping[x])
        data['Education'] = st.selectbox("Pendidikan", options=list(education_mapping.keys()), format_func=lambda x: education_mapping[x])
        data['Income'] = st.selectbox("Pendapatan", options=list(income_mapping.keys()), format_func=lambda x: income_mapping[x])
        # Tombol submit
        submit = st.form_submit_button(label='Prediksi Diabetes Saya!')
    return data, submit

# ------------------- NAVIGASI HALAMAN ------------------------------
pages = ["Prediksi Diabetes", "Visualisasi Data & Edukasi", "Tentang Model", "Tentang Dataset"]
page = st.sidebar.selectbox("📂 Navigasi Halaman", pages)

# ------------------- LOAD MODEL DAN DATASET -------------------------
model_handler = ModelHandler()
model_handler.load('models/random_forest_model.pkl', 'models/scaler.pkl')
df = pd.read_csv('data/diabetes_012_health_indicators_BRFSS2015.csv')

# =================== 1. HALAMAN PREDIKSI DIABETES ==================
if page == "Prediksi Diabetes":
    st.markdown("""
        <div style='text-align: center;'>
            <h1>SICERDIK</h1>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: center;'>        
            <h3>Sistem Cegah Risiko Diabetes berbasis Indikator Kesehatan</h3>
        </div>   
    """, unsafe_allow_html=True)
    st.write("Selamat datang di SICERDIK, sebuah aplikasi edukatif yang membantu Anda memprediksi risiko diabetes berdasarkan data kesehatan pribadi. Aplikasi ini dirancang untuk memberikan gambaran awal tentang kemungkinan risiko diabetes yang Anda miliki, sekaligus memberikan edukasi tentang cara pencegahan dan langkah hidup sehat. Silakan lengkapi data kesehatan Anda sesuai kondisi sebenarnya melalui formulir di sidebar. Mari bersama-sama melakukan deteksi dini untuk mencegah komplikasi dan menjaga kualitas hidup yang lebih baik!")

    user_data, submit = user_input_features_form()
    user_pasien = Pasien(user_data)
    user_df = user_pasien.to_dataframe()

    # Jika tombol submit diklik
    if submit:
        pred, proba = model_handler.predict(user_df)
        kelas = int(pred[0])
        st.markdown(f"<h2>Hasil Prediksi: <span style='color:#FF8303'>{target_mapping[kelas]}</span></h2>", unsafe_allow_html=True)
        st.markdown(f"#### Peluang (Probabilitas):")
        for idx, label in target_mapping.items():
            st.write(f"- {label}: {proba[0][idx]*100:.1f}%")

        # Saran otomatis edukasi sesuai hasil
        if kelas == 2:
            st.warning("⚠️ Risiko diabetes tinggi! Segera konsultasi ke dokter & mulai gaya hidup sehat: diet seimbang, olahraga rutin, kontrol berat badan.")
        elif kelas == 1:
            st.info("🧡 Anda di fase prediabetes. Kurangi gula, perbanyak aktivitas fisik, pantau berat badan & tekanan darah.")
        else:
            st.success("🥳 Risiko diabetes rendah. Tetap jaga pola makan, aktif bergerak, tidur cukup, dan hindari rokok.")

        # Simpan history prediksi
        hasil = {**user_data, 'Prediksi': target_mapping[kelas]}
        st.session_state['history'].append(hasil)

        st.divider()        
        st.title("Artikel Edukasi: Kenali Diabetes & Cara Mencegahnya")

        # Pengertian
        st.subheader("1. Pengertian Diabetes")
        st.markdown("""
        Diabetes adalah penyakit kronis yang ditandai oleh tingginya kadar gula dalam darah. Glukosa, atau gula darah, sejatinya merupakan sumber energi utama bagi tubuh. Namun, pada penderita diabetes, glukosa tidak dapat digunakan secara efektif oleh tubuh.
        Hormon insulin, yang dihasilkan oleh pankreas, berperan penting dalam mengatur kadar gula darah. Insulin membantu sel-sel tubuh menyerap glukosa agar kadar gula tetap normal. Pada penderita diabetes, pankreas tidak mampu memproduksi insulin secara memadai, atau tubuh tidak dapat memanfaatkan insulin secara optimal. Akibatnya, glukosa menumpuk dalam darah dan menimbulkan berbagai gangguan kesehatan. Jika tidak ditangani dengan baik, diabetes dapat menimbulkan komplikasi serius.
        """)

        # Jumlah Kasus
        st.subheader("2. Jumlah Kasus Diabetes di Dunia")
        st.markdown("""
        Diperkirakan sekitar 422 juta orang di seluruh dunia menderita diabetes, sebagian besar tinggal di negara berpenghasilan rendah dan menengah. Setiap tahunnya, sekitar 1,5 juta kematian disebabkan oleh diabetes. Jumlah kasus dan prevalensi penyakit ini terus meningkat dalam beberapa dekade terakhir.
        """)

        # Jenis Diabetes
        st.subheader("3. Jenis Diabetes")
        st.markdown("""
        Terdapat dua jenis utama diabetes, yaitu diabetes tipe 1 dan diabetes tipe 2.
        **Diabetes Tipe 1** terjadi ketika sistem kekebalan tubuh keliru menyerang dan menghancurkan sel-sel pankreas penghasil insulin. Akibatnya, penderita diabetes tipe 1 harus mendapatkan suplai insulin dari luar tubuh secara rutin. Kondisi ini umumnya terdiagnosis pada anak-anak atau dewasa muda.
        Sementara itu, **diabetes melitus tipe 2** terjadi akibat resistensi insulin atau produksi insulin yang tidak memadai. Kondisi ini memengaruhi cara tubuh menggunakan glukosa sebagai energi. Diabetes tipe 2 adalah bentuk diabetes yang paling umum dan sering terjadi pada orang dewasa, meskipun dapat pula dialami anak-anak atau remaja.
        """)

        # Gejala
        st.subheader("4. Gejala Diabetes")
        st.markdown("""
        Gejala diabetes tipe 1 biasanya muncul dengan cepat, dalam hitungan minggu atau bahkan hari, seperti sering buang air kecil, rasa haus berlebih, rasa lapar terus-menerus, penurunan berat badan tanpa sebab jelas, perubahan penglihatan, dan kelelahan.
        Berbeda dengan itu, diabetes tipe 2 sering kali tidak menimbulkan gejala spesifik sehingga banyak orang tidak menyadari dirinya menderita diabetes selama bertahun-tahun. Karena sifatnya yang sering diam-diam, diabetes tipe 2 dikenal sebagai “silent killer.” Gejalanya mirip dengan tipe 1, tetapi lebih samar. Oleh karena itu, penting untuk mewaspadai faktor risikonya.
        """)

        # Pengobatan dan Pencegahan
        st.subheader("5. Pengobatan dan Pencegahan Diabetes")
        st.markdown("""
        Hingga saat ini, diabetes tipe 1 belum dapat dicegah. Penanganan medis, seperti terapi insulin, bertujuan menjaga kadar gula darah tetap stabil agar gejala tidak semakin parah.
        Berbeda dengan tipe 1, diabetes tipe 2 dapat dicegah dengan pola hidup sehat, seperti meningkatkan konsumsi sayur dan buah, membatasi asupan gula, garam, dan lemak (GGL), serta rutin melakukan aktivitas fisik. Penelitian di Eropa menemukan bahwa menambah konsumsi sayur dan buah sebanyak 66 gram per hari dapat menurunkan risiko diabetes hingga 25 persen.
        Kementerian Kesehatan RI menganjurkan batas konsumsi harian sebanyak 50 gram gula (setara 4 sendok makan), 5 gram garam (setara 1 sendok teh), dan 67 gram lemak (setara 5 sendok makan). Membatasi makanan dan minuman kemasan tinggi GGL, serta mengganti camilan dengan buah atau sayur, adalah langkah efektif.
        Selain itu, aktivitas fisik minimal 30 menit per hari, atau 150 menit per minggu, sangat bermanfaat. Selain meningkatkan sensitivitas insulin, olahraga membantu menjaga berat badan ideal, memperbaiki kualitas tidur, mengatur tekanan darah, serta membuat suasana hati lebih baik.
        Diabetes dapat dicegah melalui gaya hidup sehat dan deteksi dini. Jika Anda mengalami gejala seperti di atas, segeralah berkonsultasi ke dokter untuk mendapat penanganan yang tepat.
        """)

        # Sumber Referensi
        st.markdown("""
        **Sumber Referensi:**
        - [Alodokter - Diabetes](https://www.alodokter.com/diabetes)
        - [Biofarma - Diabetes: Gejala, Penyebab, dan Pencegahan](https://www.biofarma.co.id/id/announcement/detail/diabetes-gejala-penyebab-dan-pencegahan)
        """)

        st.markdown("### Rangkuman ")
        st.markdown("""
        - **Apa itu Diabetes?** Diabetes adalah penyakit kronis akibat tubuh kesulitan mengontrol kadar gula darah.
        - **Gejala:** Sering haus, buang air kecil, berat badan turun tanpa sebab, mudah lelah.
        - **Pencegahan:** Gaya hidup sehat: makan makanan bergizi, rajin olahraga, hindari rokok & alkohol.
        - **Prediabetes:** Kondisi sebelum diabetes, masih bisa dicegah jika mengubah pola hidup.
        - **Tips:** Jaga berat badan ideal, cek kesehatan rutin, tidur cukup, kelola stres.
        """, unsafe_allow_html=True)

        # Fitur Simulasi: Jika semua data seperti user
        def simulate_user_on_dataset(model_handler, df, user_data):
            df_simulasi = df.copy()
            for col in user_data:
                df_simulasi[col] = user_data[col]
            X_simulasi = df_simulasi.drop(columns=['Diabetes_012'])
            X_simulasi_scaled = model_handler.scaler.transform(X_simulasi)
            y_simulasi_pred = model_handler.model.predict(X_simulasi_scaled)
            import numpy as np
            unique, counts = np.unique(y_simulasi_pred, return_counts=True)
            return dict(zip(unique, counts)), len(y_simulasi_pred)

        if st.button("Simulasi: Jika Semua Data Seperti Anda"):
            result_dict, total = simulate_user_on_dataset(model_handler, df, user_data)
            st.markdown(f"**Simulasi: Jika seluruh dataset punya data seperti Anda, hasil prediksi model:**")
            labels = [target_mapping.get(k, k) for k in result_dict.keys()]
            sizes = [v/total for v in result_dict.values()]
            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['#ffe5c2', '#ffba66', '#FF8303'])
            st.pyplot(fig)
            for k, v in result_dict.items():
                st.markdown(f"- {target_mapping.get(k, k)}: {v} dari {total} data ({v/total*100:.1f}%)")

    # History prediksi selama sesi
    if len(st.session_state['history']) > 0:
        st.divider()
        st.markdown("### 🕓 Riwayat Prediksi Anda Selama Sesi Ini")
        hist_df = pd.DataFrame(st.session_state['history'])
        st.dataframe(hist_df)
        csv = hist_df.to_csv(index=False).encode()
        st.download_button("⬇️ Unduh Riwayat Prediksi (CSV)", data=csv, file_name="riwayat_prediksi.csv", mime="text/csv")

# =============== 2. VISUALISASI DATA & EDUKASI ================
elif page == "Visualisasi Data & Edukasi":
    st.markdown("<h1>Visualisasi Data & Edukasi Diabetes</h1>", unsafe_allow_html=True)
    st.subheader("🟠 Distribusi Data Diabetes pada Dataset")
    plot_target_distribution(df['Diabetes_012'], target_mapping)

    st.subheader("🟠 Distribusi BMI pada Dataset")
    user_bmi = st.session_state['history'][-1]['BMI'] if st.session_state['history'] else 25
    plot_bmi_histogram_with_user(df, user_bmi)

    st.subheader("🟠 Korelasi Fitur dengan Target")
    plot_corr_with_target(df)

    st.subheader("🟠 Fitur Penting dalam Prediksi Diabetes")
    plot_feature_importance(model_handler.model, df.drop(columns=['Diabetes_012']).columns)

# =============== 3. TENTANG MODEL ===============
elif page == "Tentang Model":
    st.markdown("<h1>Tentang Model Machine Learning</h1>", unsafe_allow_html=True)
    st.markdown("""
    - **Model:** Random Forest Classifier (100 pohon keputusan) 
    - **Handling Data Tidak Seimbang:** SMOTE (Synthetic Minority Over-sampling Technique) 
    - **Feature Scaling:** StandardScaler untuk standarisasi fitur numerik 
    - **Evaluasi Model:** Stratified train/test split, classification report, confusion matrix
    """)
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    X = df.drop(columns=['Diabetes_012'])
    y = df['Diabetes_012']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test_scaled = model_handler.scaler.transform(X_test)
    y_pred = model_handler.model.predict(X_test_scaled)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("🟧 Confusion Matrix")
    plot_confusion_matrix(cm, list(target_mapping.values()))
    st.subheader("🟧 Classification Report")
    st.json(report)
    st.divider()
    st.markdown("### Kenapa Random Forest?")
    st.markdown("""
    Random Forest sangat cocok untuk data tabular medis, mampu menangani banyak fitur, dan efektif mengatasi masalah data yang tidak seimbang.
    """)
    st.divider()
    st.markdown("### ⚠️ Disclaimer")
    st.info("Aplikasi ini hanya untuk edukasi dan deteksi awal. Selalu konsultasikan ke tenaga medis untuk diagnosis & penanganan lebih lanjut.")

# =============== 4. TENTANG DATASET ===============
elif page == "Tentang Dataset":
    st.markdown("<h1>Tentang Dataset</h1>", unsafe_allow_html=True)
    st.markdown("""
    Dataset yang digunakan pada aplikasi ini adalah **Diabetes 012 Health Indicators Dataset (BRFSS 2015)**,  
    diunduh dari [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset).  
    Dataset ini merupakan hasil survei kesehatan di Amerika Serikat, berisi indikator gaya hidup, kesehatan, dan status diabetes.
    """)

    # Ringkasan dataset
    st.markdown("### ℹ️ Informasi Singkat Dataset")
    st.write(f"**Jumlah data:** {df.shape[0]:,} baris")
    st.write(f"**Jumlah fitur:** {df.shape[1]-1} fitur input + 1 target (`Diabetes_012`)")
    st.write("**Kelas Target:**")
    for key, value in target_mapping.items():
        st.write(f"- {key}: {value}")

    # Proporsi kelas
    st.markdown("### 📊 Distribusi Kelas Target")
    plot_target_distribution(df['Diabetes_012'], target_mapping)

    # Tabel contoh data
    st.markdown("### 🧾 Contoh Data")
    st.dataframe(df.sample(8, random_state=2))

    st.markdown("### 📋 Daftar Fitur & Penjelasan")
    st.markdown("""
    | Fitur | Penjelasan |
    |-------|------------|
    | HighBP | Riwayat tekanan darah tinggi (1=Ya, 0=Tidak) |
    | HighChol | Riwayat kolesterol tinggi (1=Ya, 0=Tidak) |
    | CholCheck | Pernah cek kolesterol (1=Ya, 0=Tidak) |
    | BMI | Body Mass Index (angka 12-60) |
    | Smoker | Pernah merokok (1=Ya, 0=Tidak) |
    | Stroke | Pernah mengalami stroke (1=Ya, 0=Tidak) |
    | HeartDiseaseorAttack | Riwayat jantung (1=Ya, 0=Tidak) |
    | PhysActivity | Fisik aktif/olahraga (1=Ya, 0=Tidak) |
    | Fruits | Konsumsi buah harian (1=Ya, 0=Tidak) |
    | Veggies | Konsumsi sayur harian (1=Ya, 0=Tidak) |
    | HvyAlcoholConsump | Konsumsi alkohol berat (1=Ya, 0=Tidak) |
    | AnyHealthcare | Ada akses kesehatan (1=Ya, 0=Tidak) |
    | NoDocbcCost | Tidak ke dokter karena biaya (1=Ya, 0=Tidak) |
    | GenHlth | Penilaian kesehatan umum (1=Baik, 5=Buruk) |
    | MentHlth | Hari mental kurang sehat (0-30) |
    | PhysHlth | Hari fisik kurang sehat (0-30) |
    | DiffWalk | Sulit berjalan (1=Ya, 0=Tidak) |
    | Sex | Jenis kelamin (0=Pria, 1=Wanita) |
    | Age | Kategori umur (1=18-24, dst, lihat sidebar) |
    | Education | Tingkat pendidikan (1=Tidak lulus SD, dst, lihat sidebar) |
    | Income | Pendapatan (1=<$10K, dst, lihat sidebar) |
    | Diabetes_012 | Target (0=Tidak Diabetes, 1=Prediabetes, 2=Diabetes) |
    """)

    st.info("""
    Dataset ini cocok digunakan untuk edukasi machine learning & kesehatan publik.  
    Namun, hasil prediksi pada aplikasi ini **bukan diagnosis medis**, selalu konsultasi ke dokter untuk penanganan lebih lanjut.
    """)


# =============== FOOTER ======================
st.markdown("<hr><center><span style='color:#FF8303'>Dibuat untuk edukasi kesehatan dan pembelajaran machine learning oleh Hafizh Iman Wicaksono.</span></center>", unsafe_allow_html=True)
