import numpy as np
import pickle
import streamlit as st

### Loading the model
with open('MODEL.pkl', 'rb') as file:
    model = pickle.load(file)
file.close()

st.title("Salary Prediction Using Machine Learning")
Years_Of_Experince = st.number_input('Years_Of_Experince')
if st.button('Predict'):
    x = np.array([[Years_Of_Experince]])
    Y_predict = model.predict(x)
    st.success('Prediction is {}'.format(Y_predict))