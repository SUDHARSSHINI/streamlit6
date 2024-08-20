import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from responsibleai import RAIInsights, FeatureMetadata
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Load the dataset
data = pd.read_csv('heart_disease_prediction.csv')

# Prepare the dataset
X = data.drop(columns=['Heart Disease'])
y = data['Heart Disease']

# Define categorical features
categorical_features = ['Sex']

# Convert categorical columns to string if they are not already
X[categorical_features] = X[categorical_features].astype(str)

# Check and handle missing values
if X.isnull().any().any() or y.isnull().any():
    X = X.dropna()
    y = y[X.index]  # Align y with the cleaned X

# Create a column transformer to preprocess the data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and model training
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(random_state=42))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Define a function to predict heart disease from user input
def predict_heart_disease(age, sex, cholesterol, blood_pressure):
    new_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })
    new_data['Sex'] = new_data['Sex'].astype(str)

    predictions = pipeline.predict(new_data)
    return predictions[0]

# Streamlit app
st.title("Heart Disease Prediction")

# Get user input
age = st.number_input("Enter age:", min_value=20, max_value=100, step=1)
sex = st.selectbox("Select sex:", ['0', '1'])
cholesterol = st.number_input("Enter cholesterol level:", min_value=150, max_value=300, step=1)
blood_pressure = st.number_input("Enter blood pressure:", min_value=80, max_value=180, step=1)

if st.button('Predict'):
    result = predict_heart_disease(age, sex, cholesterol, blood_pressure)
    st.write("Predicted Heart Disease Status:", "Yes" if result == 1 else "No")
