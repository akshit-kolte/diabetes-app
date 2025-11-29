import pandas as pd
import numpy as np
import joblib
import streamlit as st
#Load the Model

model=joblib.load(open("logistic_regression_model(1).joblib",'rb'))

st.title("Diabetes app")
#Input feature
Pregnancies=st.number_input("Pregnancies",min_value=0.0)
Glucose=st.number_input("Glucose",min_value=0.0)
BloodPressure=st.number_input("BloodPressure",min_value=0.0)
SkinThickness=st.number_input("SkinThickness",min_value=0.0)
Insulin=st.number_input("Insulin",min_value=0.0)
BMI=st.number_input("BMI",min_value=0.0)
DiabetesPedigreeFunction=st.number_input("DiabetesPedigreeFunction",min_value=0.0)
Age=st.number_input("Age",min_value=0.0)

#Make Pred
if st.button('Predict Diabetes'):
	input_data=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
	prediction=model.predict(input_data)[0]
	st.success(f'Predict Diabetes:{prediction:.2f}')
