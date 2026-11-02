#!/usr/bin/env python3
"""
softmax_regression.py

Multiclass (softmax) logistic regression implemented from scratch in NumPy,
applied to cuisine classification from recipe ingredients.

Model:   P = softmax(X W),  X: (N, D) sparse TF-IDF,  W: (D, C)
Loss:    L(W) = -(1/N) * sum(Y * log P) + mu * ||W||^2
Grad:    dL/dW = X^T (P - Y) / N + 2 * mu * W
Update:  W <- W - eta * dL/dW       (full-batch gradient descent)

Usage:
    python src/softmax_regression.py                 # submitted settings
    python src/softmax_regression.py --mu 1e-5 --eta 5 --iters 3000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


class SoftmaxRegression:
    def __init__(self, eta=0.4, mu=1e-3, max_iter=5000):
        self.eta, self.mu, self.max_iter = eta, mu, max_iter

    def _loss(self, X, Y, W):
        P = softmax(X @ W, axis=1)
        return -np.sum(Y * np.log(P + 1e-15)) / X.shape[0] + self.mu * np.sum(W * W)

    def _grad(self, X, Y, W):
        P = softmax(X @ W, axis=1)
        return X.T @ (P - Y) / X.shape[0] + 2 * self.mu * W

    def fit(self, X, y, log_every=500):
        self.classes_, y_idx = np.unique(np.asarray(y), return_inverse=True)
        Y = np.eye(len(self.classes_))[y_idx]               # one-hot (N, C)
        W = np.zeros((X.shape[1], Y.shape[1]))
        self.loss_history_ = []
        for step in range(1, self.max_iter + 1):
            W -= self.eta * self._grad(X, Y, W)
            self.loss_history_.append(self._loss(X, Y, W))
            if log_every and step % log_every == 0:
                print(f"step {step:5d}  loss {self.loss_history_[-1]:.4f}")
        self.W_ = W
        return self

    def predict_proba(self, X):
        return softmax(X @ self.W_, axis=1)

    def predict(self, X):
        return self.classes_[np.argmax(X @ self.W_, axis=1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--eta", type=float, default=0.4)
    ap.add_argument("--mu", type=float, default=1e-3)
    ap.add_argument("--iters", type=int, default=5000)
    args = ap.parse_args()

    train = pd.read_json(Path(args.data) / "train.json")
    clean = lambda ings: [i.lower().strip() for i in ings]

    # each ingredient phrase (e.g. "olive oil") is one TF-IDF token
    vec = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x,
                          token_pattern=None, min_df=1)
    X = vec.fit_transform(train["ingredients"].apply(clean))
    y = train["cuisine"].values
    print("Feature matrix:", X.shape)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = SoftmaxRegression(eta=args.eta, mu=args.mu, max_iter=args.iters).fit(X_tr, y_tr)
    pred = model.predict(X_val)
    print(f"train acc {accuracy_score(y_tr, model.predict(X_tr)):.4f} | "
          f"val acc {accuracy_score(y_val, pred):.4f} | "
          f"val macro-F1 {f1_score(y_val, pred, average='macro'):.3f}")


if __name__ == "__main__":
    main()
