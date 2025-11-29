import pandas as pd
import numpy as np
import joblib
import streamlit as st
#Load the Model

model=joblib.load(open("logistic_regression_model (1).joblib", 'rb'))

st.title("Diabetes Prediction app")
#Input feature
Age=st.number_input("Age of the person",min_value=0.0)
BMI=st.number_input("BMI",min_value=0.0)
BloodPressure=st.number_input("Newspaper Adv Budget",min_value=0.0)
Insulin=st.number_input("Insulin",min_value=0.0)
Glucose=st.number_input("Glucose",min_value=0.0)
SkinThickness=st.number_input("SkinThickness",min_value=0.0)
DiabetesPedigreeFunction=st.number_input("DiabetesPedigreeFunction",min_value=0.0)
Pregnancies=st.number_input("Pregnancies",min_value=0.0)

#Make Pred
if st.button('Diabetes Prediction'):
	input_data=np.array([[Age,BMI,BloodPressure,Insulin,Glucose,SkinThickness,DiabetesPedigreeFunction,Pregnancies]])
	prediction_model=model.predict(input_data)[0]
	st.success(f'Predict Diabetes:{prediction:.2f}')
