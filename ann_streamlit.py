import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle


# Loading the trained model
model = tf.keras.models.load_model('mymodel.keras')

# Loading pickle files
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('ohe_geo.pkl', 'rb') as file:
    ohe_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scalar = pickle.load(file)


# --------------------------------------------------------
# Streamlit 
#---------------------------------------------------------

st.title("Customer Churn Prediction")

#Streamlit User Inputs
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 20, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
est_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_prods = st.slider('Number of Products', 1,4)
has_cr_card = st.selectbox('Has Credit Card', ['No', 'Yes'])
isActive_mem = st.selectbox('Is active member', ['No', 'Yes'])

if has_cr_card == 'No':
    has_cr_card_num = 0
else:
    has_cr_card_num = 1

if isActive_mem == 'No':
    isActive_mem_num = 0
else:
    isActive_mem_num = 1

#Preparing input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_prods],
    'HasCrCard': [has_cr_card_num],
    'IsActiveMember': [isActive_mem_num],
    'EstimatedSalary': [est_salary]
})


#One HOt Encoding Geography
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_endoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

#Adding Geo Info in Input Data
input_data = pd.concat([input_data.reset_index(drop=True), geo_endoded_df], axis=1)

#Scaling Data
input_data_scaled = scalar.transform(input_data)

#Prediction Churn
prediction = model.predict(input_data_scaled)
pred_prob = prediction[0][0]

st.write(f'Churn Probability: {pred_prob:.2f}')

if pred_prob > 0.5:
    st.write ("The customer is likely to be churn")
else:
    st.write("The customer is not likely to be churn")
