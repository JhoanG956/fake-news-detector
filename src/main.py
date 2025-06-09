import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import clean_text
from src.vectorization import get_tfidf_features
from src.models import train_and_evaluate

df = pd.read_csv("dataset/training_data.csv")
df["clean_text"] = df["title"].apply(clean_text)

X, vectorizer = get_tfidf_features(df["clean_text"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_and_evaluate(X_train, X_test, y_train, y_test, model_type="lr")