import streamlit as st
import joblib
import numpy as np
import pandas as pd
import xgboost
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.connectors import PandasConnector
import os

# Load Model ML, minmax dan kolom
def load_ml_model():
    model = joblib.load('XGBoost_best_model.pkl')
    scaler = joblib.load('minmax_scaler.pkl')
    column_names = joblib.load('column_names.pkl')
    return model, scaler, column_names

# Page prediksi
def ml_prediction_page():
    model, scaler, column_names = load_ml_model()

    st.write("## Prediksi Biaya Total Pasien dengan Machine Learning")

    # widgets
    drug_quantity = st.number_input('Jumlah Obat:', value=1)
    days_diff = st.number_input('Lama hari:', value=1.0, format="%.1f")

    age_group = st.selectbox('Umur:', ['0-18', '19-30', '31-45', '46-60', '60+'])
    age_mapping = {'0-18': 0, '19-30': 1, '31-45': 2, '46-60': 3, '60+': 4}
    age_group_value = age_mapping[age_group]

    room_type_encoded = st.selectbox('Tipe Kamar:', ['Tidak Digunakan', 'Kelas 3', 'Kelas 2', 'Kelas 1', 'VIP'])
    room_mapping = {'Tidak Digunakan': 0, 'Kelas 3': 1, 'Kelas 2': 2, 'Kelas 1': 3, 'VIP': 4}
    room_type_value = room_mapping[room_type_encoded]

    branch = st.selectbox('Cabang RS Siloam:', ['RSMA', 'RSMD', 'RSMS'])
    branch_values = {'branch_RSMA': 0, 'branch_RSMD': 0, 'branch_RSMS': 0}
    branch_values[branch] = 1

    gender = st.selectbox('Jenis kelamin:', ['Laki-laki', 'Perempuan'])
    gender_value = 1 if gender == 'Laki-laki' else 0

    payment = st.selectbox('Metode Pembayaran:', ['Asuransi', 'Pribadi'])
    payment_value = 1 if payment == 'Asuransi' else 0

    hospital_care = st.selectbox('Tipe perawatan rumah sakit:', ['Rawat Inap', 'Rawat Jalan'])
    hospital_care_value = 1 if hospital_care == 'Rawat Inap' else 0

    drug_brand = st.selectbox('Merk Obat:', ['Blackmores', 'Calpol', 'Diclofenac', 'Enervon-C', 'Holland & Barrett', 
                                             'Naproxen', 'Panadol', 'Paramex', 'Tramadol'])
    drug_brand_values = {f'drug_brand_{brand}': 0 for brand in ['Blackmores', 'Calpol', 'Diclofenac', 'Enervon-C', 
                                                                'Holland & Barrett', 'Naproxen', 'Panadol', 'Paramex', 
                                                                'Tramadol']}
    drug_brand_values[f'drug_brand_{drug_brand}'] = 1

    drug_type = st.selectbox('Tipe obat:', ['Pereda Nyeri', 'Umum', 'Vitamin'])
    drug_type_values = {f'drug_type_{type}': 0 for type in ['Pereda Nyeri', 'Umum', 'Vitamin']}
    drug_type_values[f'drug_type_{drug_type}'] = 1

    doctor = st.selectbox('Dokter:', ['Penyakit Dalam', 'Umum'])
    doctor_values = {f'doctor_{doc}': 0 for doc in ['Penyakit Dalam', 'Umum']}
    doctor_values[f'doctor_{doctor}'] = 1

    lab = st.selectbox('Uji Lab:', ['Hematologi', 'Kimia Darah', 'Serologi'])
    lab_values = {f'lab_{test}': 0 for test in ['Hematologi', 'Kimia Darah', 'Serologi']}
    lab_values[f'lab_{lab}'] = 1

    # semua input
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

    # Prediki
    if st.button("Predict"):
        predictions = model.predict(new_input_scaled)
        st.write("Predictions:", predictions[0])

# fungsi untuk LLM interaction page
def llm_interaction_page():
    # inisialisasi ai
    llm_t0 = OpenAI(
        api_token=os.getenv("OPENAI_API_KEY"),
        temperature=0,
        seed=26
    )

    # Field descriptions
    field_descriptions = {
        'id': 'The unique identifier for each hospital visit',
        'date_in': 'The date when the patient was admitted to the hospital',
        'date_out': 'The date when the patient was discharged from the hospital',
        'branch': 'The branch of the hospital where the patient was admitted',
        'hospital_care': 'Type of care provided to the patient (e.g., inpatient, outpatient)',
        'payment': 'The payment method used by the patient (e.g., insurance, private)',
        'review': 'Patient satisfaction review ranging from very satisfied to very dissatisfied',
        'patient_name': 'The name of the patient',
        'gender': 'The gender of the patient',
        'age': 'The age of the patient',
        'room_type': 'The type of room the patient stayed in during hospital care',
        'drug_brand': 'The brand of medication provided to the patient',
        'drug_type': 'The type of drug administered to the patient (e.g., general medicine, antibiotic)',
        'doctor': 'The specialist doctor attending to the patient (e.g., surgeon, pediatrician)',
        'surgery': 'Details about any surgery performed, with categories like minor, major, none',
        'lab': 'The type of laboratory tests performed (e.g., hematology, serology)',
        'is_DBD': 'A boolean indicating whether the patient was diagnosed with dengue fever'
    }

    # Load dataset
    df = pd.read_csv('mini_data_rev.csv')

    #  Buat PandasConnector and SmartDataframe
    connector = PandasConnector({"original_df": df}, field_descriptions=field_descriptions)
    sdf = SmartDataframe(connector, config={"llm": llm_t0, "enable_cache": False})

    st.write("## Tanya Bit AI")

    st.write("""
    Ini merupakan bagian percobaan jika dibuat bentuk web dan untuk file csv nya menggunakan csv yang sudah disederhanakan.
    """)

    # Display logo Bithealth
    logo_path = 'Website_Logo_Bithealth_.png'
    st.image(logo_path, use_column_width=True)

    # Text input user query
    user_query = st.text_input("Ingin tahu apa?")

    def display_plot(image_path):
        """Display the image in Streamlit."""
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()
            st.image(image_bytes)

    if st.button("Kirim"):
        if user_query:
            try:
                # get response  PandasAI
                response = sdf.chat(user_query)
                if isinstance(response, str) and response.endswith('.png'):
                    display_plot(response)
                else:
                    st.write("Jawaban:")
                    st.write(response)
            except Exception as e:
                st.write(f"error: {e}")
        else:
            st.write("Masukan instruksi atau pertanyaan mengenai data RS.")

# Main app utama
st.set_page_config(page_title="Tes LLM Final Project Kelompok 01", page_icon="Website_Logo_Bithealth_.png")

st.title("Tes LLM Final Project Kelompok 01")

st.write("""
Ini merupakan halaman utama. Silakan pilih salah satu dari opsi berikut:
""")

# Sidebar
page = st.sidebar.radio("Pilih Halaman", ("Prediksi Biaya Total dengan ML", "Tanya Bit AI"))

# pilihan page
if page == "Prediksi Biaya Total dengan ML":
    ml_prediction_page()
elif page == "Tanya Bit AI":
    llm_interaction_page()
