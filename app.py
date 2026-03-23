import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Customer Segmentation App")

fresh = st.number_input("Fresh")
milk = st.number_input("Milk")
grocery = st.number_input("Grocery")
frozen = st.number_input("Frozen")
detergents = st.number_input("Detergents_Paper")
delicassen = st.number_input("Delicassen")

if st.button("Predict Cluster"):
    data = np.array([[fresh, milk, grocery, frozen, detergents, delicassen]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)

    st.success(f"Customer belongs to Cluster: {result[0]}")