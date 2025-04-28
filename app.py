# app.py

import streamlit as st
import pickle
import numpy as np

# Load model and scaler
with open('model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    scaler = data['scaler']

# List of feature names
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.title("🔬 Breast Cancer Prediction App")
st.write("Enter the tumor features below:")

# Collect user inputs dynamically
user_inputs = []

# Layout: Two columns
col1, col2 = st.columns(2)

for idx, feature in enumerate(feature_names):
    with (col1 if idx % 2 == 0 else col2):
        value = st.number_input(f"{feature.replace('_', ' ').capitalize()}", value=0.0)
        user_inputs.append(value)

# Prediction
if st.button('Predict'):
    features = np.array(user_inputs).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    if prediction[0] == 1:
        st.success('❌ The tumor is **Malignant (Cancer Positive)**')
    else:
        st.success('✅ The tumor is **Benign (Cancer Negative)**')
