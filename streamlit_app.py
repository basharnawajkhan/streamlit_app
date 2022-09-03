import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

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

data = pd.read_csv("C:/Users/BASHAR/PycharmProjects/pythonProject1/telecommunications_churn.csv")

from sklearn.ensemble import IsolationForest
df = data.copy()
clf = IsolationForest(random_state=10, contamination=0.1)
clf.fit(df)
outliere_predict = clf.predict(df)
print(outliere_predict)

df["Score"] = clf.decision_function(df)
df["Anomaly"] = clf.predict(df.iloc[:, 0:19])

outliers = df[df["Anomaly"]==-1]
df1 = df.drop(df.index[df['Anomaly'] == -1]).reset_index()
df1 = df1.drop(columns=["index"])
df2 = df1.iloc[:, 0:19]

df3 = df2[["voice_mail_plan", "voice_mail_messages", "international_mins", "customer_service_calls", "international_plan", "international_calls", "international_charge", "total_charge", "churn"]]

X = df3.iloc[:,:-1]
Y = df3.iloc[:,-1]

No_churn = df3[df3['churn']==0]
Churn = df3[df3['churn']==1]

from sklearn.utils import resample
from collections import Counter

over_sampling = resample(Churn, replace=True, n_samples=len(No_churn), random_state=42)
over_sample = pd.concat([No_churn, over_sampling])

final_df = over_sample.reset_index()
final_df = final_df.drop(["index"], axis=1)

from sklearn.model_selection import train_test_split

X = final_df.iloc[:,:-1]
Y = final_df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, shuffle=True)

def norm_func (i):
    x = (i - i.min())/ (i.max()-i.min())
    return (x)

x_train = norm_func(x_train)
x_test =  norm_func(x_test)

from sklearn.svm import SVC
Model = SVC(kernel="rbf", random_state=7, gamma=100)
Model.fit(x_train, y_train)
Prediction = Model.predict(result)

st.subheader("Predicted Result")
st.write("loyal Customer" if Prediction[0]== 0 else "Churn Customer")