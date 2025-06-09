import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Descargar recursos si no están
nltk.download("stopwords")

# --- Preprocesamiento ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Cargar datos ---
df = pd.read_csv("dataset/training_data.csv", sep="\t", header=None, names=["label", "title"])
print("Primeras filas del dataset:")
print(df.head())
df["clean_text"] = df["title"].apply(clean_text)

# --- Vectorización ---
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# --- Dividir en train/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Entrenar modelo ---
model = LogisticRegression()
model.fit(X_train, y_train)

# --- Evaluar ---
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Guardar modelo y vectorizador ---
joblib.dump(model, "modelo_entrenado.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Modelo y vectorizador guardados")