import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from responsibleai import RAIInsights, FeatureMetadata
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric

# Load the dataset
def load_data():
    return pd.read_csv('/content/heart_disease_prediction.csv')

# Function to load or train the model
def load_or_train_model():
    data = load_data()

    # Define features and target
    X = data.drop(columns=['Heart Disease'])
    y = data['Heart Disease']
    
    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    # Train a RandomForest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, sc, X_test, y_test

# Function to initialize Responsible AI insights
def initialize_rai_insights(model, X_test, y_test):
    test_df = pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='Heart Disease')], axis=1)
    
    # Convert to AIF360 dataset format
    test_data_aif360 = BinaryLabelDataset(
        df=test_df,
        label_names=['Heart Disease'],
        protected_attribute_names=['Sex']
    )
    
    # Initialize FeatureMetadata
    feature_metadata = FeatureMetadata(
        categorical_features=['Sex']
    )
    
    # Initialize RAIInsights
    rai_insights = RAIInsights(
        model=model,
        train=pd.concat([pd.DataFrame(X_test), pd.Series(y_test, name='Heart Disease')], axis=1),
        test=test_df,
        target_column='Heart Disease',
        task_type='classification',
        feature_metadata=feature_metadata
    )
    
    # Add components
    rai_insights.explainer.add()
    rai_insights.error_analysis.add()
    rai_insights.counterfactual.add(total_CFs=10, desired_class='opposite')
    rai_insights.causal.add(treatment_features=['Cholesterol', 'Blood Pressure'])
    
    # Compute insights
    rai_insights.compute()
    
    return rai_insights

# Load model and scaler
model, scaler, X_test, y_test = load_or_train_model()

# Initialize Responsible AI insights
rai_insights = initialize_rai_insights(model, X_test, y_test)

# Streamlit App
st.title('Heart Disease Prediction')

# User input
age = st.number_input("Enter age:", min_value=18, max_value=100, step=1)
sex = st.selectbox("Enter sex (0 for female, 1 for male):", [0, 1])
cholesterol = st.number_input("Enter cholesterol level:", min_value=150, max_value=300, step=1)
blood_pressure = st.number_input("Enter blood pressure:", min_value=80, max_value=180, step=1)

# Predict heart disease
def predict_heart_disease(age, sex, cholesterol, blood_pressure):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'Cholesterol': [cholesterol],
        'Blood Pressure': [blood_pressure]
    })
    
    # Scale input data
    input_data_scaled = scaler.transform(input_data)
    
    # Predict using the model
    prediction = model.predict(input_data_scaled)
    return 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

if st.button('Predict'):
    prediction_result = predict_heart_disease(age, sex, cholesterol, blood_pressure)
    st.write(f"The predicted status is: {prediction_result}")

# Display Responsible AI insights
st.header("Responsible AI Insights")

# Display fairness metrics
metric = ClassificationMetric(rai_insights.test_data, rai_insights.counterfactual.get(), privileged_groups=[{'Sex': 1}], unprivileged_groups=[{'Sex': 0}])

st.subheader("Fairness Metrics")
st.write("Disparate Impact:", metric.disparate_impact())
st.write("Statistical Parity Difference:", metric.statistical_parity_difference())
st.write("Equal Opportunity Difference:", metric.equal_opportunity_difference())
st.write("Average Odds Difference:", metric.average_odds_difference())

# Display Responsible AI insights components
st.subheader("Explainer Insights")
st.write(rai_insights.explainer.get())

st.subheader("Error Analysis Insights")
st.write(rai_insights.error_analysis.get())

st.subheader("Counterfactual Insights")
st.write(rai_insights.counterfactual.get())

st.subheader("Causal Analysis Insights")
st.write(rai_insights.causal.get())
