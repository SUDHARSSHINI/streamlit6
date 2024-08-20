import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from responsibleai import ResponsibleAI
from responsibleai.metrics import MetricType

# Load the dataset
data = pd.read_csv('/content/heart_disease_prediction.csv')

# Define features and target
X = data.drop('Heart Disease', axis=1)
y = data['Heart Disease']

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f'Accuracy: {accuracy:.2f}')

# Responsible AI analysis
def analyze_responsible_ai(model, X_train, y_train, X_test, y_test):
    r = ResponsibleAI(model=model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    r.add_metric(MetricType.FAIRNESS)
    r.add_metric(MetricType.ACCURACY)
    r.add_metric(MetricType.PREDICTIONS)
    r.run()
    return r

# Analyze the model
rai_analysis = analyze_responsible_ai(model, X_train, y_train, X_test, y_test)

# Display Responsible AI properties
st.write("Responsible AI Analysis:")
st.write(f"Fairness Metrics: {rai_analysis.metrics['Fairness']}")
st.write(f"Accuracy Metrics: {rai_analysis.metrics['Accuracy']}")
st.write(f"Predictions Metrics: {rai_analysis.metrics['Predictions']}")

# Define a function to predict heart disease from user input
def predict_heart_disease_from_input():
    st.title("Heart Disease Prediction")

    # Get user input for each feature using Streamlit's widgets
    age = st.number_input("Enter the patient's age (0-100):", min_value=0, max_value=100, step=1)
    sex = st.radio("Enter the patient's sex:", (1, 0))  # Assume 1: Male, 0: Female
    cholesterol = st.number_input("Enter the patient's cholesterol level:")
    blood_pressure = st.number_input("Enter the patient's blood pressure:")
    
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })

    # Scale the features
    input_data = sc.transform(input_data)

    # Predict heart disease
    prediction = model.predict(input_data)
    prediction_human_readable = np.where(prediction == 1, 'Heart Disease', 'No Heart Disease')

    # Display the prediction
    st.write(f"The predicted heart disease status is: {prediction_human_readable[0]}")

# Call the function to test
predict_heart_disease_from_input()
