# Cuisine Classification with Multiclass Logistic Regression

Predicting a recipe's cuisine (20 classes) from its ingredient list, using **softmax logistic regression implemented from scratch in NumPy**, benchmarked against scikit-learn's SAGA solver.

Course project, **AIL7024 – Machine Learning**, IIT Delhi (Oct–Nov 2025), guided by Prof. Tanmoy Chakraborty.
Team: Adarsh Pandey (2025JRB2021), Pritosh Kumar.

## Problem

- **Train:** 29,774 recipes labelled with one of 20 cuisines · **Test:** 10,000 unlabelled recipes
- Input is a variable-length list of ingredient strings (avg. 10.8 per recipe)
- Strong class imbalance: Italian 19.8%, Mexican 16.1%, Southern US 11.0% … Brazilian 1.15%

## Approach

**Features: sparse TF-IDF over ingredients**
- From-scratch model: each ingredient phrase (e.g. `"olive oil"`) is a single token → 6,203 features
- SAGA baseline: ingredients normalized to single tokens (`olive_oil`), unigram + bigram TF-IDF, top 20k features

**Model: softmax regression from scratch (`src/softmax_regression.py`)**

```
P    = softmax(X W)                                  X: N×D sparse, W: D×C
L(W) = -(1/N) Σ Y ⊙ log P  +  μ ‖W‖²                 cross-entropy + L2
∇L   = Xᵀ(P − Y) / N  +  2μW
W   ← W − η ∇L                                       full-batch gradient descent
```

Fully vectorized, operating directly on the sparse TF-IDF matrix.

**Baseline:** `sklearn.linear_model.LogisticRegression(solver="saga")` (`src/cuisine_prediction_saga.py`).

## Results (stratified 80/20 split, 5,955 validation recipes)

| Model | Settings | Val. accuracy | Macro F1 |
|---|---|---|---|
| Softmax regression (NumPy, as submitted) | η=0.4, μ=1e-3, 5000 iters | 56.5% | 0.26 |
| Softmax regression (NumPy, tuned) | η=5, μ=1e-5, 3000 iters | 75.5% | 0.66 |
| scikit-learn SAGA (uni+bigram TF-IDF) | max_iter=2000 | **73.4%** | 0.62 |

SAGA per-class F1: Mexican 0.88, Indian 0.82, Italian 0.80, Chinese 0.79 (best); British 0.27, Irish 0.33 (worst, due to overlapping European ingredients and few samples).

The submitted from-scratch configuration was over-regularized: with L2-normalized TF-IDF rows, μ=1e-3 keeps the weights too small and the model underfits (train accuracy ≈ validation ≈ 57%). Reducing μ to 1e-5 lets the same NumPy implementation reach 75.5% validation accuracy.

## Repository

```
notebooks/cuisine_logreg_from_scratch.ipynb   original course notebook (EDA, model, test predictions)
src/softmax_regression.py                     standalone from-scratch implementation
src/cuisine_prediction_saga.py                scikit-learn SAGA baseline + submission
reports/                                      project report and presentation slides
results/                                      test-set predictions (10,000 rows each)
```

## Run

```bash
pip install numpy scipy pandas scikit-learn joblib
# put train.json and test.json in data/
python src/softmax_regression.py                          # submitted settings
python src/softmax_regression.py --mu 1e-5 --eta 5 --iters 3000
python src/cuisine_prediction_saga.py
```
