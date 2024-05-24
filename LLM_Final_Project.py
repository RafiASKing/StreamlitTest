import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from pandasai.connectors import PandasConnector

# Initialize the OpenAI LLM
llm_t0 = OpenAI(
    api_token="sk-proj-0NA2KRfDMuRpqPlfWXRZT3BlbkFJSwQyGg5HuFfd3jWH8CZC",
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

# Load the dataset
df = pd.read_csv('C:/Users/RafiWangsaSeniang/Desktop/STREAM/mini_data_rev.csv')

# Create the PandasConnector and SmartDataframe
connector = PandasConnector({"original_df": df}, field_descriptions=field_descriptions)
sdf = SmartDataframe(connector, config={"llm": llm_t0, "enable_cache": False})

# Streamlit App
st.set_page_config(page_title="Tes LLM Final Project Kelompok 01", page_icon="C:/Users/RafiWangsaSeniang/Desktop/STREAM/Website_Logo_Bithealth_.png")

st.title("Tes LLM Final Project Kelompok 01")

st.write("""
Ini merupakan bagian percobaan jika dibuat bentuk web dan untuk file csv nya menggunakan csv yang sudah disederhanakan.
""")

# Display the logo
logo_path = 'C:/Users/RafiWangsaSeniang/Desktop/STREAM/Website_Logo_Bithealth_.png'
st.image(logo_path, use_column_width=True)

# Text input for user query
user_query = st.text_input("Ingin tahu apa?")

def display_plot(image_path):
    """Display the image in Streamlit."""
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
        st.image(image_bytes)

if st.button("Kirim"):
    if user_query:
        try:
            # Get the response from PandasAI
            response = sdf.chat(user_query)
            if isinstance(response, str) and response.endswith('.png'):
                display_plot(response)
            else:
                st.write("Jawaban:")
                st.write(response)
        except Exception as e:
            st.write(f"An error occurred: {e}")
    else:
        st.write("Please enter a query.")
