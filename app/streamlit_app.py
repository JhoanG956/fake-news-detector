import re
import string
import streamlit as st
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


st.title("Detector de Noticias Falsas")

model = joblib.load("modelo_entrenado.pkl")
vectorizer = joblib.load("vectorizer.pkl")

input_text = st.text_input("Ingresa un titular de noticia:")
if input_text:
    clean_input = clean_text(input_text)
    vectorized_input = vectorizer.transform([clean_input])
    prediction = model.predict(vectorized_input)[0]
    st.write("Resultado:", "✅ Real" if prediction == 1 else "❌ Falsa")