"""
Stratified K-Fold Cross-Validation for TF-IDF + Linear SVM
Computes accuracy, precision, recall, F1, ROC-AUC across folds.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

def wordopt(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["class"] = 0
    true["class"] = 1
    df = pd.concat([fake, true], axis=0, ignore_index=True)
    df = df.drop(["title", "subject", "date"], axis=1)
    df["text"] = df["text"].astype(str).apply(wordopt)
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    X = df["text"].values
    y = df["class"].values
    return X, y

def run_cv(n_splits: int = 5, random_state: int = 42):
    X, y = load_data()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs, f1s, aucs = [], [], []

    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        vectorizer = TfidfVectorizer()
        xv_train = vectorizer.fit_transform(X_train)
        xv_test = vectorizer.transform(X_test)

        base_svm = LinearSVC(max_iter=10000, random_state=random_state)
        clf = CalibratedClassifierCV(base_svm)
        clf.fit(xv_train, y_train)

        y_pred = clf.predict(xv_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)

        # AUC using predict_proba if available; else decision_function
        try:
            probs = clf.predict_proba(xv_test)[:, 1]
            auc = roc_auc_score(y_test, probs)
        except Exception:
            try:
                scores = clf.decision_function(xv_test)
                auc = roc_auc_score(y_test, scores)
            except Exception:
                auc = np.nan

        accs.append(acc)
        f1s.append(f1)
        aucs.append(auc)

        print(f"Fold {fold}: accuracy={acc:.4f}  f1={f1:.4f}  roc_auc={auc if not np.isnan(auc) else 'N/A'}")

    print("\nSummary (Stratified K-Fold)")
    print(f"Accuracy: mean={np.mean(accs):.4f}  std={np.std(accs):.4f}")
    print(f"F1-score: mean={np.mean(f1s):.4f}  std={np.std(f1s):.4f}")
    if not all(np.isnan(aucs)):
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        print(f"ROC-AUC:  mean={np.mean(valid_aucs):.4f}  std={np.std(valid_aucs):.4f}")
    else:
        print("ROC-AUC:  N/A")

if __name__ == "__main__":
    run_cv()


