import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.connectors import PandasConnector
from pandasai.helpers.openai_info import get_openai_callback
from datetime import datetime
import os

# Set Konfigurasi Page
st.set_page_config(page_title="Tes LLM Final Project Kelompok 01", page_icon="Website_Logo_Bithealth_.png")

# Load CSS untuk estetika
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")  # Memastikan .css dalam directory yang sama

# Function to load the ML model and scaler
@st.cache_resource
def load_ml_model():
    model = joblib.load('XGBoost_best_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    column_names = joblib.load('column_names.pkl')
    return model, scaler, column_names

# Fungsi untuk kalkulasi days_diff untuk rawat inap
def calculate_days_diff(date_in, date_out):
    if date_in >= date_out:
        st.error("Tanggal keluar harus setelah tanggal masuk.")
        return None
    return (date_out - date_in).days + 1

# Fungsi untuk ML prediction page
def ml_prediction_page():
    model, scaler, column_names = load_ml_model()

    st.write("## Prediksi Biaya Total dengan ML")

    # Input patient name
    patient_name = st.text_input('Nama Pasien:', value="", help="Masukkan nama pasien")

    # Organize input ke sections tertentu
    st.write("### Informasi Pasien")
    age_group = st.selectbox('Umur:', ['0-18', '19-30', '31-45', '46-60', '60+'], help="Pilih kelompok umur pasien")
    age_mapping = {'0-18': 0, '19-30': 1, '31-45': 2, '46-60': 3, '60+': 4}
    age_group_value = age_mapping[age_group]
    gender = st.selectbox('Jenis kelamin:', ['Laki-laki', 'Perempuan'], help="Pilih jenis kelamin pasien")
    gender_value = 1 if gender == 'Laki-laki' else 0
    branch = st.selectbox('Cabang RS Siloam:', ['RSMA', 'RSMD', 'RSMS'], help="Pilih cabang rumah sakit")
    branch_values = {'branch_RSMA': 0, 'branch_RSMD': 0, 'branch_RSMS': 0}
    branch_values[branch] = 1
    payment = st.selectbox('Metode Pembayaran:', ['Asuransi', 'Pribadi'], help="Pilih metode pembayaran")
    payment_value = 1 if payment == 'Asuransi' else 0

    st.write("### Informasi Perawatan")
    hospital_care = st.radio('Tipe perawatan rumah sakit:', ['Rawat Inap', 'Rawat Jalan'], help="Pilih tipe perawatan")
    if hospital_care == 'Rawat Inap':
        date_in = st.date_input('Tanggal Masuk', value=datetime.today(), help="Pilih tanggal masuk")
        date_out = st.date_input('Tanggal Keluar', value=datetime.today(), help="Pilih tanggal keluar")
        days_diff = calculate_days_diff(date_in, date_out)
    else:
        days_diff = 0.0

    room_type_encoded = st.selectbox('Tipe Kamar:', ['Tidak Digunakan', 'Kelas 3', 'Kelas 2', 'Kelas 1', 'VIP'], help="Pilih tipe kamar")
    room_mapping = {'Tidak Digunakan': 0, 'Kelas 3': 1, 'Kelas 2': 2, 'Kelas 1': 3, 'VIP': 4}
    room_type_value = room_mapping[room_type_encoded]
    hospital_care_value = 1 if hospital_care == 'Rawat Inap' else 0

    st.write("### Informasi Obat dan Dokter")
    drug_brand = st.selectbox('Merk Obat:', ['Blackmores', 'Calpol', 'Diclofenac', 'Enervon-C', 'Holland & Barrett', 
                                             'Naproxen', 'Panadol', 'Paramex', 'Tramadol'], help="Pilih merk obat")
    drug_brand_values = {f'drug_brand_{brand}': 0 for brand in ['Blackmores', 'Calpol', 'Diclofenac', 'Enervon-C', 
                                                                'Holland & Barrett', 'Naproxen', 'Panadol', 'Paramex', 
                                                                'Tramadol']}
    drug_brand_values[f'drug_brand_{drug_brand}'] = 1
    drug_type = st.selectbox('Tipe obat:', ['Pereda Nyeri', 'Umum', 'Vitamin'], help="Pilih tipe obat")
    drug_type_values = {f'drug_type_{type}': 0 for type in ['Pereda Nyeri', 'Umum', 'Vitamin']}
    drug_type_values[f'drug_type_{drug_type}'] = 1
    drug_quantity = st.number_input('Jumlah Obat:', value=1, help="Masukkan jumlah obat")
    doctor = st.selectbox('Dokter:', ['Penyakit Dalam', 'Umum'], help="Pilih dokter")
    doctor_values = {f'doctor_{doc}': 0 for doc in ['Penyakit Dalam', 'Umum']}
    doctor_values[f'doctor_{doctor}'] = 1
    lab = st.selectbox('Uji Lab:', ['Hematologi', 'Kimia Darah', 'Serologi'], help="Pilih uji lab")
    lab_values = {f'lab_{test}': 0 for test in ['Hematologi', 'Kimia Darah', 'Serologi']}
    lab_values[f'lab_{lab}'] = 1

    # Collect input datas
    new_data = {
        'drug_quantity': drug_quantity,
        'days_diff': days_diff,
        'age_group': age_group_value,
        'room_type_encoded': room_type_value,
        'branch_RSMA': branch_values['branch_RSMA'],
        'branch_RSMD': branch_values['branch_RSMD'],
        'branch_RSMS': branch_values['branch_RSMS'],
        'hospital_care_Rawat Inap': hospital_care_value,
        'payment_Asuransi': payment_value,
        'gender_Laki-laki': gender_value,
        **drug_brand_values,
        **drug_type_values,
        **doctor_values,
        **lab_values
    }

    # Convert ke DataFrame
    new_input_df = pd.DataFrame([new_data])
    new_input_df = new_input_df.reindex(columns=column_names, fill_value=0)

    # Scale input data
    new_input_scaled = scaler.transform(new_input_df)

    # Predict
    if st.button("Predict"):
        if days_diff is not None:
            predictions = model.predict(new_input_scaled)
            formatted_prediction = f"Rp {predictions[0]:,.0f}"
            patient_name_display = patient_name if patient_name else "ini"
            st.write(f"Biaya yang pasien {patient_name_display} perlu bayar ialah sebanyak <b>{formatted_prediction}</b>", unsafe_allow_html=True)
        else:
            st.write("Pastikan input tanggal valid, lalu coba lagi!")

# Function untuk display image
def display_image(image_path, width=300, center=False):
    if center:
        st.image(image_path, width=width, use_column_width='auto')
    else:
        st.image(image_path, width=width)

# Function to load the LLM interaction page
def llm_interaction_page():
    # Initialize OpenAI LLM
    llm_t0 = OpenAI(api_token=os.getenv("OPENAI_API_KEY"), temperature=0, seed=26)

    # Load dataset
    @st.cache_data
    def load_data():
        return pd.read_csv('mini_data_rev.csv')
    
    df = load_data()

    # Create PandasConnector and SmartDataframe
    connector = PandasConnector({"original_df": df})
    sdf = SmartDataframe(connector, config={"llm": llm_t0, "conversational": False})

    st.write("## Tanya BitAI")

    st.write("""
    Ini merupakan fitur eksperimental menggunakan LLM untuk query dan bertanya mengenai data rumah sakit.
    """)

    # Display new logo
    new_logo_path = 'PandaLogoLLM.png'
    st.image(new_logo_path, caption='Powered by OpenAI GPT-3.5', width=300)

    user_query = st.text_input("Kamu dapat query dan bertanya mengenai data rumah sakit disini", key="user_query")

    def display_plot(image_path):
        """Display the image in Streamlit."""
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            st.image(image_bytes)

    if st.button("Kirim"):
        if user_query:
            try:
                # Get the response from PandasAI
                with get_openai_callback() as cb:
                    response = sdf.chat(user_query)
                    if isinstance(response, str) and response.endswith('.png'):
                        display_plot(response)
                    else:
                        st.write("**Jawaban:**")
                        st.write(response)
                        st.write("================================================")
                        st.write(f"Tokens terpakai: {cb.total_tokens}")
                        st.write(f"Total Cost dalam (USD): ${cb.total_cost:.6f}")
            except Exception as e:
                if "context_length_exceeded" in str(e):
                    st.error("Pertanyaan Anda terlalu panjang. Harap perpendek pertanyaan Anda dan coba lagi.")
                else:
                    st.error(f"Terjadi kesalahan: {e}")
        else:
            st.error("Masukkan pertanyaan terlebih dahulu.")

# Function homepage
def homepage():
    st.title("Deploy App Final Project Kelompok 01 Hactiv8")

    # Display main page
    main_logo_path = 'Website_Logo_Bithealth_.png'
    display_image(main_logo_path, width=300)

    st.write("""
    **Disusun oleh:**
    - Arya Sangga Buana (Data Trainee)
    - Nadya Novahina (Data Trainee) 
    - Rafi Athallah Seniang (Data Trainee)
    - Rosita Laili Udhiah (Data Trainee)
    """)

    st.write("""
    ### Tentang Proyek Ini
    App ini merupakan bagian dari Final Project kami di program training BitHealth GradXpert Academy 2024 dengan objektif untuk meningkatkan digitalisasi dan manajemen data di rumah sakit. App ini mencakup beberapa komponen utama:

    1. **Prediksi Biaya Total dengan Machine Learning**:
        - Menggunakan model machine learning untuk memprediksi biaya total perawatan di rumah sakit dari input data pasien oleh user.
        - Input dari user mencakup informasi demografis, jenis perawatan, tipe kamar, merk dan jumlah obat, serta informasi dokter dan uji lab.
        - Prediksi biaya berpotensi membantu manajemen rumah sakit dalam membuat keputusan finansial dan operasional yang lebih baik.

    2. **Tanya BitAI**:
        - Fitur interaktif yang memungkinkan user untuk mengajukan pertanyaan terkait data transaksi pasien menggunakan Language Learning Model (LLM).
        - Dapat memberikan jawaban berdasarkan analisis data yang tersedia, membantu dalam mendapatkan insight yang berharga dari data pasien.

    ### Latar Belakang Proyek
    Transformasi digital di industri kesehatan sangat penting untuk meningkatkan kualitas layanan dan efisiensi operasional rumah sakit. Dalam upaya ini, data yang terintegrasi dan analisis yang tepat menjadi kunci untuk mencapai tujuan tersebut. Project ini dirancang untuk:
    - Mengotomatiskan pengumpulan dan pembersihan data.
    - Menganalisis data untuk mendapatkan informasi dan insight yang berguna.
    - Membuat prediksi machine learning untuk membantu manajemen rumah sakit dalam pengambilan keputusan finansial dan operasional.

    ### Manfaat Proyek
    - **Bagi Pasien**: Meningkatkan transparansi biaya perawatan dan layanan yang diberikan oleh rumah sakit.
    - **Bagi Manajemen Rumah Sakit**: Memberikan tools untuk analisis data yang lebih baik dan prediksi biaya yang akurat, sehingga membantu dalam perencanaan dan pengambilan keputusan.
    - **Bagi Industri Kesehatan**: Mendorong penerapan teknologi digital yang lebih luas dan efisien, serta meningkatkan kualitas layanan kesehatan secara keseluruhan.

    ### Tim Kami
    - Arya Sangga Buana (Data Trainee)
    - Nadya Novahina (Data Trainee)
    - Rafi Athallah Seniang (Data Trainee)
    - Rosita Laili Udhiah (Data Trainee)

    ### Petunjuk Penggunaan
    Gunakan navigasi di sebelah kiri untuk berpindah antara halaman prediksi biaya dan halaman interaksi dengan BitAI. Pada halaman prediksi biaya, Anda dapat melakukan input data pasien untuk mendapatkan estimasi biaya perawatan. Pada halaman Tanya BitAI, Anda dapat mengajukan pertanyaan terkait data transaksi pasien dan mendapatkan jawaban berdasarkan analisis data BitAI.

    ### Dokumentasi dan Proposal
    Untuk informasi lebih lanjut, Anda dapat mengakses dokumentasi dan proposal lengkap di [link berikut](https://docs.google.com/document/d/1Yb0k6IiaAAe92uFUsXmtk8nuISqJr1PEWoz7wSWZt8s/edit#heading=h.16323hskb57r).

    ### Sumber
    Data transaksi pasien yang ada dan kami gunakan diprovide oleh Hactiv8.
    """)

# Function for the help page
def help_page():
    st.write("""
    ## Bantuan
    Di halaman ini, Anda dapat menemukan panduan dan tips untuk menggunakan fitur Tanya BitAI serta Prediksi Biaya Total dengan Machine Learning.
      
    # Machine Learning
    **Petunjuk Penggunaan Prediksi Biaya Total**:
    - Masukkan informasi pasien dan perawatan yang diminta pada form yang tersedia.
    - Pastikan semua input diisi dengan benar dan lengkap.
    - Tekan tombol "Predict" untuk mendapatkan estimasi biaya perawatan.
    - Jika terdapat kesalahan atau input tidak valid, periksa kembali data yang dimasukkan.

    **Contoh Input untuk Prediksi Biaya Total**:
    - Nama Pasien: "Budi"
    - Umur: "19-30"
    - Jenis Kelamin: "Laki-laki"
    - Cabang RS Siloam: "RSMA"
    - Metode Pembayaran: "Asuransi"
    - Tipe Perawatan: "Rawat Jalan"
    - Tipe Kamar: "VIP"
    - Merk Obat: "Panadol"
    - Tipe Obat: "Pereda Nyeri"
    - Jumlah Obat: 5
    - Dokter: "Penyakit Dalam"
    - Uji Lab: "Kimia Darah"

    # LLM BitAI
    **Petunjuk Penggunaan Tanya BitAI**:
    - Masukkan pertanyaan Anda di kotak teks yang tersedia.
    - Tekan tombol "Kirim" untuk mendapatkan jawaban dari sistem.
    - Jika mengalami masalah, pastikan pertanyaan Anda tepat dan relevan dengan data transaksi pasien.

    **Contoh Pertanyaan untuk Tanya BitAI**:
    - "Berapa usia rata-rata pasien?"
    - "Berapa pasien rawat inap pada bulan Desember 2023?"
    - "Obat apa yang paling sering digunakan?"

    Untuk pertanyaan lebih lanjut, hubungi tim kami melalui email.

    Email: Kelompok01Data@bithealth.co.id
    """)

# Main app
# Sidebar untuk navigasi
page = st.sidebar.radio("Pilih Halaman", ("Beranda", "Prediksi Biaya Total dengan ML", "Tanya BitAI", "Bantuan"))

# Display page pilihan
if page == "Beranda":
    homepage()
elif page == "Prediksi Biaya Total dengan ML":
    ml_prediction_page()
elif page == "Tanya BitAI":
    llm_interaction_page()
elif page == "Bantuan":
    help_page()

# Feedback section
st.sidebar.header("Feedback")
feedback = st.sidebar.text_area("Masukkan feedback Anda di sini:")
if st.sidebar.button("Kirim feedback"):
    st.sidebar.write("Terima kasih atas feedback Anda!")
