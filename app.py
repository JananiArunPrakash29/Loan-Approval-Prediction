import streamlit as st 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier

st.title("Loan Approval Prediction")

data=pd.read_csv("loan_data.csv")

x=data[["Annual_Income"]]
y=data["Loan_Status"]

model=DecisionTreeClassifier(random_state=42)
model.fit(x,y) 

income=st.number_input("Enter Annual Income", min_value=0)

if st.button("predict"):
    result=model.predict(pd.DataFrame({"Annual_Income":[income]}))

    if result[0]=="Approved":
        st.success("Loan Approved!")
    else:
        st.error("Loan Not Approved!")
        st.write("Reason:Annual income does not meet the model's learned approval criteria.")