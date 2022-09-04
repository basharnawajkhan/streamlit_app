import numpy as np
import pandas as pd
import pickle
import streamlit as st

classifier = pickle.load(open("trained_model.pkl", "rb"))

st.title("Telecommunications Churn Prediction")
st.sidebar.header("User Input Parameters")

def user_input_features():
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ("1", "0"))
    voice_mail_messages	 = st.sidebar.number_input("Number voice mail messages")
    international_mins = st.sidebar.number_input("Total international mins")
    customer_service_calls = st.sidebar.number_input("Number customer service calls")
    international_plan = st.sidebar.selectbox("International Plan", ("1", "0"))
    international_calls = st.sidebar.number_input("Number international calls")
    international_charge = st.sidebar.number_input("Total international charge")
    total_charge = st.sidebar.number_input("Total charge")
    Data = {"voice_mail_plan":voice_mail_plan, "voice_mail_messages":voice_mail_messages, "international_mins":international_mins,
           "customer_service_calls":customer_service_calls, "international_plan":international_plan,
           "international_calls":international_calls, "international_charge":international_charge, "total_charge":total_charge}
    features = pd.DataFrame(Data, index=[0])
    return features

result = user_input_features()

st.subheader("User Input Parameters")
st.write(result)

def prediction(result):
    prediction = classifier.predict()
    print (prediction)
    return prediction

st.subheader("Predicted Result")
st.write(prediction)

