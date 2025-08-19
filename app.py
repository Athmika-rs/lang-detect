import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and vectorizer
model = joblib.load("language_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page config
st.set_page_config(page_title="🌍 Language Detection App", page_icon="🌐", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 12px;
    }
    .stTextInput > div > div > input {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #e8f5e9;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        color: #2e7d32;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🌍 Language Detection App")
st.write("Detect the **language** of any sentence using **TF-IDF + Random Forest**.")

# User Input
user_input = st.text_input("✍️ Enter a sentence:")

if st.button("🔍 Detect Language"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a sentence to detect.")
    else:
        # Transform input
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(input_tfidf)[0]

        # Display result in styled box
        st.markdown(f"<div class='result-box'> Predicted Language: {prediction} </div>", unsafe_allow_html=True)

# Optional: Show Confusion Matrix (expandable)
with st.expander("📊 Model Performance (Confusion Matrix)"):
    try:
        # Load dataset for performance check
        df = pd.read_csv("Language_Dataset.csv")
        X = df["text"]
        y = df["language"]

        X_vec = vectorizer.transform(X)
        y_pred = model.predict(X_vec)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, y_pred, labels=model.classes_)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
    except Exception as e:
        st.error("Confusion matrix could not be generated. Make sure dataset is uploaded.")
