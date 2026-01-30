import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

model = tf.keras.models.load_model('salary_prediction_model.h5')

with open('scaler_regression.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_geo.pkl', 'rb') as f:
    one_hot_encoder = pickle.load(f)

st.title("Salary Prediction App")

geography = st.selectbox("Select Geography", one_hot_encoder.categories_[0])
gender = st.selectbox("Select Gender",label_encoder_gender.classes_)
age = st.slider('Age', 18, 100)
tenure = st.slider('Tenure (Years at Company)', 0, 10)
balance = st.number_input('Account Balance', min_value=0)       
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])       
is_active_member = st.selectbox("Is Active Member", [0, 1])
credit_score = st.slider('Credit Score', 300, 850)

input_data = pd.DataFrame({
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],   
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'CreditScore': [credit_score]
})
geo_encoded = one_hot_encoder.transform([[geography]])

geo_encoded_df = pd.DataFrame(
    geo_encoded.toarray(),
    columns=one_hot_encoder.get_feature_names_out(['Geography'])
)

input_data = pd.concat([input_data, geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)
predicted_salary = prediction[0][0]

st.subheader(f'Predicted Salary: ${predicted_salary:,.2f}')





