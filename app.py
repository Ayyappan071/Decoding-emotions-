# app.py

import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load model and vectorizer
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

def preprocess(text):
    text = re.sub(r"http\S+|@\S+|#\S+", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.lower().split()
    return " ".join(tokens)

st.title("Emotion Detection from Text")

user_input = st.text_area("Enter a sentence to analyze emotion:")

if st.button("Analyze"):
    cleaned = preprocess(user_input)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)
    st.success(f"Predicted Emotion: **{prediction[0]}**")
