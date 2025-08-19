import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("Language_Dataset.csv")

X = df["text"]
y = df["language"]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_vec, y)

st.title("🌍 Language Detection App")

user_input = st.text_input("Enter a sentence:")

if user_input:
    input_vec = vectorizer.transform([user_input])
    prediction = model.predict(input_vec)[0]
    st.success(f"Predicted Language: **{prediction}**")
