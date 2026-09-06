"""
Flask web app for the Spam Email Classifier.

Loads the pre-trained TF-IDF vectorizer + Naive Bayes / SVM models from
model/ (trained by src/train.py) and serves a small UI + JSON API to
classify pasted email text with either model.

Run locally:
    pip install -r requirements.txt
    python src/train.py     # only needed once, to (re)build model/*.pkl
    python app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import sys

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_PATH = os.path.join(BASE_DIR, "data", "emails.csv")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

app = Flask(__name__)


def _ensure_models_exist():
    """Train on first run if model/*.pkl haven't been generated yet."""
    needed = ["vectorizer.pkl", "naive_bayes.pkl", "svm.pkl"]
    if all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in needed):
        return
    if not os.path.exists(DATA_PATH):
        import generate_data
        generate_data.main()
    import train
    train.main()


_ensure_models_exist()

vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
models = {
    "naive_bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "svm": joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
}
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    METRICS = json.load(f)

# Vocabulary each model associates most strongly with spam - used to
# highlight "why" a message was flagged. Built once at startup.
feature_names = vectorizer.get_feature_names_out()


def top_spam_terms(model_name, limit=200):
    model = models[model_name]
    # MultinomialNB exposes feature_log_prob_; the calibrated SVM wraps
    # LinearSVC estimators inside calibrated_classifiers_.
    try:
        if model_name == "naive_bayes":
            classes = list(model.classes_)
            spam_idx = classes.index("spam")
            ham_idx = classes.index("ham")
            log_ratio = model.feature_log_prob_[spam_idx] - model.feature_log_prob_[ham_idx]
            ranked = log_ratio.argsort()[::-1][:limit]
        else:
            base = model.calibrated_classifiers_[0].estimator
            classes = list(base.classes_)
            spam_idx = classes.index("spam")
            coefs = base.coef_[0] if len(classes) == 2 else base.coef_[spam_idx]
            ranked = coefs.argsort()[::-1][:limit]
        return set(feature_names[i] for i in ranked)
    except Exception:
        return set()


SPAM_TERMS = {name: top_spam_terms(name) for name in models}


def flagged_words(text, model_name):
    words = set(w.strip(".,!?:;\"'()").lower() for w in text.split())
    hits = [w for w in words if w in SPAM_TERMS.get(model_name, set())]
    return sorted(hits)[:10]


@app.route("/")
def index():
    return render_template("index.html", metrics=METRICS)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/classify", methods=["POST"])
def api_classify():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    model_name = payload.get("model", "naive_bayes")

    if not text:
        return jsonify({"error": "Paste some email text first."}), 400
    if model_name not in models:
        return jsonify({"error": f"Unknown model '{model_name}'."}), 400

    model = models[model_name]
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    confidence = float(proba[classes.index(prediction)])

    return jsonify({
        "label": prediction,
        "confidence": round(confidence, 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "flagged_words": flagged_words(text, model_name),
        "model": model_name,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
