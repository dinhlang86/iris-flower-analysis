import numpy as np
import streamlit as st
from predict_page import svm_model, scaler, le

st.title('Iris Flower Species Prediction')
st.write('''This app predicts the specie of an Iris Flower''')
st.write('''### Input parameters''')
sepal_length = st.slider('Sepal Length', min_value=5.0, max_value=8.0)
sepal_width = st.slider('Sepal Width', min_value=2.0, max_value=5.0)
petal_length = st.slider('Petal Length', min_value=1.0, max_value=7.0)
petal_width = st.slider('Petal Width', min_value=0.0, max_value=3.0)
ok = st.button('Check Flower Specie')

if ok:
    features = np.array(
        [[sepal_length, sepal_width, petal_length, petal_width]])
    features = scaler.transform(features)
    prediction = svm_model.predict(features)
    st.subheader(
        f'The specie of the iris flower is: {le.classes_[prediction[0]]}')
