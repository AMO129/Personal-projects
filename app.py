import streamlit as st
import joblib
import pandas as pd

# Load the brain you exported
model = joblib.load('salary_predictor_model.pkl')
model_cols = joblib.load('model_columns.pkl')

st.title("Salary Growth Predictor")

# The inputs the user will see
dept = st.selectbox("Role", ["IT", "HR", "Finance", "Unknown"])
current_exp = st.number_input("Current Years of Experience", 0, 40, 5)
years_to_add = st.slider("Years in the future to predict?", 0, 10, 0)

if st.button("Predict Salary"):
    # Calculate future experience
    future_exp = current_exp + years_to_add
    
    # Create the row of data for the model
    # Note: We use 30 as a default age and 160 as default hours
    input_df = pd.DataFrame([[30 + years_to_add, future_exp, 160]], 
                            columns=['Age', 'YearsExperience', 'MonthlyHoursWorked'])
    
    # Standardize the categories (The 1s and 0s)
    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    if f'Department_{dept}' in model_cols:
        input_df[f'Department_{dept}'] = 1
        
    # Reorder columns and predict
    prediction = model.predict(input_df[model_cols])
    st.success(f"Predicted Monthly Salary: ${prediction[0]:,.2f}")