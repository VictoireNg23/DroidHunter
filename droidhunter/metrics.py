"""Metrics of the paper: accuracy / precision / recall / F1 (Tables 3-4), AUT (Table 5)."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y, pred, score=None):
    y, pred = np.asarray(y), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y.tolist())) == 2
    return {
        "n": int(y.size), "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)) if both else None,
        "roc_auc": float(roc_auc_score(y, score)) if both and score is not None else None,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def aut(values):
    """Area Under Time (Tesseract): trapezoids over N periods, normalised to [0, 1]."""
    v = np.asarray(values, dtype=float)
    if v.size < 2:
        return None
    return float(np.sum((v[:-1] + v[1:]) / 2) / (v.size - 1))


def per_year(y, pred, score, years):
    groups = defaultdict(list)
    for i, year in enumerate(years):
        if year:
            groups[year].append(i)
    if not groups:
        return None
    by_year = {}
    for year in sorted(groups):
        idx = groups[year]
        by_year[year] = binary_metrics([y[i] for i in idx], [pred[i] for i in idx],
                                       None if score is None else [score[i] for i in idx])
    order = sorted(by_year)
    return {"years": by_year,
            "aut_accuracy": aut([by_year[k]["accuracy"] for k in order]),
            "aut_f1": aut([by_year[k]["f1"] for k in order])}
