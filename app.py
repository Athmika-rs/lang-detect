import streamlit as st
import joblib

# Load model & vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Streamlit UI
st.title("🌍 Language Detection App")
st.write("Detects the language of a given sentence using TF-IDF + Random Forest.")

# User Input
user_input = st.text_input("Enter a sentence:")

if user_input:
    # Transform input
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]
    st.success(f"Predicted Language: **{prediction}**")
