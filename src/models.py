from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_evaluate(X_train, X_test, y_train, y_test, model_type="lr"):
    if model_type == "lr":
        model = LogisticRegression()
    elif model_type == "rf":
        model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model