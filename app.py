import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pipe = pickle.load(open(os.path.join(BASE_DIR, 'pipe.pkl'), 'rb'))
columns = pickle.load(open(os.path.join(BASE_DIR, 'columns.pkl'), 'rb'))
options = pickle.load(open(os.path.join(BASE_DIR, 'options.pkl'), 'rb'))

st.title("💻 Laptop Predictor")

#INPUTS 

company = st.selectbox('Brand', options['Company'])

laptop_type = st.selectbox('Type', options['TypeName'])

ram = st.selectbox('RAM (GB)', [2,4,6,8,12,16,24,32,64])

weight = st.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=2.0)

touchscreen = st.selectbox('Touchscreen', ['No','Yes'])
ips = st.selectbox('IPS', ['No','Yes'])

screen_size = st.slider('Screen Size (inches)', 10.0, 18.0, 13.0)

resolution = st.selectbox('Resolution', [
    '1920x1080','1366x768','1600x900','3840x2160',
    '3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'
])

cpu = st.selectbox('CPU', options['Cpu brand'])

hdd = st.selectbox('HDD (GB)', [0,128,256,512,1024,2048])
ssd = st.selectbox('SSD (GB)', [0,128,256,512,1024])

gpu = st.selectbox('GPU', options['Gpu brand'])

os_name = st.selectbox('OS', options['os'])

#  PREDICTION

if st.button('Predict Price'):

    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    query = pd.DataFrame([{
        'Company': company,
        'TypeName': laptop_type,
        'Ram': ram,
        'Weight': weight,
        'Touchscreen': touchscreen,
        'Ips': ips,
        'ppi': ppi,
        'Cpu brand': cpu,
        'HDD': hdd,
        'SSD': ssd,
        'Gpu brand': gpu,
        'os': os_name
    }])

    query = query.reindex(columns=columns, fill_value=0)

    prediction = np.exp(pipe.predict(query)[0])

    st.success(f"💰 Predicted Price: ₹ {int(prediction):,}")
