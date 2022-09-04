import numpy as np
import pandas as pd
import pickle
import streamlit as st

classifier = pickle.load(open("trained_model.pkl", "rb"))


def predict_note_authentication(voice_mail_plan, voice_mail_messages, international_mins, customer_service_calls, international_plan, international_calls, international_charge, total_charge):
    """Let's Authenticate the Banks Note
    This is using docstrings for specifications.
    ---
    parameters:
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values

    """

    prediction = classifier.predict([[voice_mail_plan, voice_mail_messages, international_mins, customer_service_calls, international_plan, international_calls, international_charge, total_charge]])
    print(prediction)
    return prediction

def main():
    st.title("BTelecommunications Churn Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    voice_mail_plan = st.sidebar.selectbox("Voice Mail Plan", ("1", "0"))
    voice_mail_messages = st.sidebar.number_input("Number voice mail messages")
    international_mins = st.sidebar.number_input("Total international mins")
    customer_service_calls = st.sidebar.number_input("Number customer service calls")
    international_plan = st.sidebar.selectbox("International Plan", ("1", "0"))
    international_calls = st.sidebar.number_input("Number international calls")
    international_charge = st.sidebar.number_input("Total international charge")
    total_charge = st.sidebar.number_input("Total charge")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")


if __name__ == '__main__':
    main()

