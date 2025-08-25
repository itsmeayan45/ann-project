import streamlit as st
import numpy as np
import os

# Check if required files exist
required_files = ['model.keras', 'label_encoder_gender.pkl', 'scalar.pkl', 'onehot_encode_geo.pkl']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.error("Please make sure all model files are uploaded to your repository.")
    st.stop()

try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    import pandas as pd
    import pickle
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.error("Please check your requirements.txt file")
    st.stop()

## Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model.keras')

@st.cache_resource
def load_encoders():
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('scalar.pkl', 'rb') as file:
        scalar = pickle.load(file)
    with open('onehot_encode_geo.pkl', 'rb') as file:
        onehot_encode_geo = pickle.load(file)
    return label_encoder_gender, scalar, onehot_encode_geo

# Load models and encoders
model = load_model()
label_encoder_gender, scalar, onehot_encode_geo = load_encoders()

### Streamlit app
st.title('Customer Churn Prediction')

## User input
geography = st.selectbox('Geography', onehot_encode_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Add prediction button
if st.button('Predict Churn'):
    ## Prepare the input data
    input_data = {
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }

    ## One hot encode geography
    geo_encoded = onehot_encode_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encode_geo.get_feature_names_out(['Geography']))

    ## Create input dataframe
    input_df = pd.DataFrame(input_data)

    ## Combine one-hot encoded columns with input data
    input_df = pd.concat([input_df, geo_encoded_df], axis=1)

    ## Scale the input data
    input_data_scaled = scalar.transform(input_df)

    ## Make prediction
    with st.spinner('Making prediction...'):
        prediction = model.predict(input_data_scaled)
        prediction_prob = prediction[0][0]

    ## Display results
    st.write(f"Churn Probability: {prediction_prob:.4f}")

    ## Apply threshold condition
    if prediction_prob > 0.5:
        st.error("The customer is likely to churn")
    else:
        st.success("The customer is not likely to churn")
