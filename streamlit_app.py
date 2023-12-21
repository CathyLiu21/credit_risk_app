import streamlit as st
import numpy as np
import pickle

st.title("Credit Risk Prediction")

age = st.slider("**Select Age:**", 18, 70, 30)
sex = st.radio("**Select Gender:**", ["Male", "Female"])
job_info = """
**Select Job Type:**
- 0: unemployed/unskilled
- 1: unskilled
- 2: skilled employee
- 3: management/highly qualified employee
"""
job = st.slider(job_info, 0, 3, 1)
housing = st.radio("**Owns a House?**", ["Yes", "No"])
saving_acc="""
- little: <  100 DM
- moderate: [100, 500) DM
- quite rich: [500, 1000) DM
- rich: >= 1000 DM
"""
saving_account = st.selectbox(saving_acc, ["little", "moderate", "quite rich", "rich"])
checking_amount = st.slider("**Checking Amount (DM):**", 0, 20000, 1000)
credit_amount = st.slider("**Credit Amount:**", 0, 50000, 10000)

duration = st.slider("**Duration (Months):**", 0, 72, 12)

purpose = st.selectbox("**Purpose**", ["radio/tv", "education", "furniture/equipment", "new car", "used car", "business", "domestic appliance", "repairs", "other"])

if st.button("Predict"):
    test_arr = np.array([age, sex, job, housing, saving_account, checking_amount, credit_amount, duration, purpose])
    test_arr = np.reshape(test_arr, (1, -1))

    model = pickle.load(open('ml_model.pkl', 'rb'))
    prediction = model.predict(test_arr)
    predicted = "Risky" if prediction else "No Risk" 
    result = f"The model has predicted that the result is: {predicted}"
    st.write(result)
