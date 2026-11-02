#!/usr/bin/env python3
"""
cuisine_prediction_saga.py

Baseline: TF-IDF (ingredient tokens, unigram + bigram) + scikit-learn
multinomial Logistic Regression (SAGA solver).

- Loads data/train.json and data/test.json
- Preprocesses ingredients (lowercase, remove punctuation, join multi-word
  ingredients into single tokens, e.g. "olive oil" -> "olive_oil")
- Trains on a stratified 80/20 split and reports validation metrics
- Retrains on the full training set and writes results/submission_saga.csv

Requirements: pip install pandas scikit-learn joblib
"""
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

# ---------- Config ----------
DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "train.json"
TEST_PATH = DATA_DIR / "test.json"
OUT_DIR = Path("results")
SUBMISSION_PATH = OUT_DIR / "submission_saga.csv"
MODEL_PATH = OUT_DIR / "cuisine_model.joblib"
LE_PATH = OUT_DIR / "label_encoder.joblib"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM = (1, 2)
LOGREG_MAX_ITER = 2000


# ---------- Helpers ----------
def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_ingredient(s: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace and join
    multi-word ingredients with an underscore."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)   # remove punctuation
    s = re.sub(r"\s+", "_", s)       # join multi-word into single token
    return s


def recipe_to_text(ings):
    return " ".join(normalize_ingredient(i) for i in ings if isinstance(i, str) and i.strip())


# ---------- Main ----------
def main():
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        print(f"ERROR: put train.json and test.json in {DATA_DIR}/")
        sys.exit(1)
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    train_df = pd.DataFrame(load_json(TRAIN_PATH))
    test_df = pd.DataFrame(load_json(TEST_PATH))

    print("Preprocessing ingredients...")
    train_df["text"] = train_df["ingredients"].apply(recipe_to_text)
    test_df["text"] = test_df["ingredients"].apply(recipe_to_text)

    le = LabelEncoder()
    y = le.fit_transform(train_df["cuisine"])
    print(f"Detected {len(le.classes_)} cuisine classes.")

    X_train, X_val, y_train, y_val = train_test_split(
        train_df["text"], y, test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    print("Train/Validation sizes:", X_train.shape[0], X_val.shape[0])

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(token_pattern=r"[^ ]+",
                                  ngram_range=TFIDF_NGRAM,
                                  max_features=TFIDF_MAX_FEATURES)),
        # softmax (multinomial) loss is the default for multiclass problems
        ("clf", LogisticRegression(max_iter=LOGREG_MAX_ITER, solver="saga")),
    ])

    print("Training (on train split)...")
    pipe.fit(X_train, y_train)

    val_pred = pipe.predict(X_val)
    print(f"Validation accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print("\nClassification report (validation):")
    print(classification_report(y_val, val_pred, target_names=le.classes_))

    print("Retraining on full training set...")
    pipe.fit(train_df["text"], y)

    test_labels = le.inverse_transform(pipe.predict(test_df["text"]))
    submission = pd.DataFrame({"Id": test_df["id"], "Category": test_labels}).sort_values("Id")
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to: {SUBMISSION_PATH}")

    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(le, LE_PATH)
    print("Done.")


if __name__ == "__main__":
    main()
