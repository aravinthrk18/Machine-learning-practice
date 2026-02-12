import streamlit as st
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.title("Student Pass/Fail Prediction")

st.write("Enter student details:")

hours = st.number_input("Study Hours", 0.0, 24.0, 4.0)
attendance = st.number_input("Attendance Percentage", 0.0, 100.0, 75.0)
marks = st.number_input("Internal Marks", 0.0, 50.0, 25.0)

if st.button("Predict"):
    prediction = model.predict([[hours, attendance, marks]])
    probability = model.predict_proba([[hours, attendance, marks]])

    pass_prob = probability[0][1] * 100

    if prediction[0] == 1:
        st.success(f"Prediction: PASS (Confidence: {pass_prob:.2f}%)")
    else:
        st.error(f"Prediction: FAIL (Confidence: {100-pass_prob:.2f}%)")
