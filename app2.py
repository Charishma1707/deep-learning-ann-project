import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf


model = tf.keras.models.load_model('regression_model.h5')

# Load encoders
with open('label_encoder2_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder2_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler2.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title
st.title(' Customer Salary Prediction App')

# User Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', ['Male', 'Female'])
credit_score = st.number_input('Credit Score')
age = st.slider('Age', 18, 92)
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited=st.selectbox('Exited', [0, 1])

# Encode inputs
gender_encoded = label_encoder_gender.transform([gender])[0]
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Prepare input DataFrame
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited':[exited]
})

# Combine with one-hot geo
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale
input_data_scaled = scaler.transform(input_data)

# Predict Salary
predicted_salary = model.predict(input_data_scaled)

# If it's a 2D array, flatten it:
if isinstance(predicted_salary, np.ndarray):
    predicted_salary = predicted_salary.flatten()[0]



# Show result
st.success(f"💸 Predicted Estimated Salary: ₹{predicted_salary:,.2f}")
