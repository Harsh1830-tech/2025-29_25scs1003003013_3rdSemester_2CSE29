"""
Trains two spam classifiers on data/emails.csv:
  - Multinomial Naive Bayes  (classic, fast, great baseline for text)
  - Linear SVM (LinearSVC)   (usually edges out NB on accuracy for text)

Both share one TF-IDF vectorizer so their scores are directly comparable.
Saves everything the Flask app needs into model/:
  vectorizer.pkl, naive_bayes.pkl, svm.pkl, metrics.json

Run:
    python src/train.py
"""

import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "emails.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # --- Naive Bayes ---
    nb = MultinomialNB()
    nb.fit(X_train_vec, y_train)
    nb_preds = nb.predict(X_test_vec)

    # --- Linear SVM, wrapped so we can still get a confidence probability ---
    svm = CalibratedClassifierCV(LinearSVC(), cv=3)
    svm.fit(X_train_vec, y_train)
    svm_preds = svm.predict(X_test_vec)

    def score(preds):
        return {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, pos_label="spam"), 4),
            "recall": round(recall_score(y_test, preds, pos_label="spam"), 4),
            "f1": round(f1_score(y_test, preds, pos_label="spam"), 4),
        }

    metrics = {
        "naive_bayes": score(nb_preds),
        "svm": score(svm_preds),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_spam": int((df["label"] == "spam").sum()),
        "n_ham": int((df["label"] == "ham").sum()),
    }

    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.pkl"))
    joblib.dump(nb, os.path.join(MODEL_DIR, "naive_bayes.pkl"))
    joblib.dump(svm, os.path.join(MODEL_DIR, "svm.pkl"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("\nSaved vectorizer + models to model/")


if __name__ == "__main__":
    main()
