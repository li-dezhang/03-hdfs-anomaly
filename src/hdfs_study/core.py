"""Keyed input validation, splitting, scoring and validation-only thresholds."""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, confusion_matrix

SEED = 20261021

def merge_data(matrix, labels):
    for frame in (matrix, labels):
        if frame.BlockId.isna().any() or frame.BlockId.duplicated().any():
            raise ValueError('Missing or duplicate BlockId')
    if set(matrix.BlockId) != set(labels.BlockId):
        raise ValueError('Feature/label ID sets differ')
    if labels.Label.isna().any() or not labels.Label.isin(['Normal', 'Anomaly']).all():
        raise ValueError('Missing/unknown label')
    features = [c for c in matrix if re.fullmatch(r'E[1-9][0-9]*', c)]
    features.sort(key=lambda c: int(c[1:]))
    if not features:
        raise ValueError('No event features')
    joined = matrix.merge(labels.rename(columns={'Label': 'truth'}), on='BlockId', validate='one_to_one')
    joined = joined.sort_values('BlockId').reset_index(drop=True)
    x = joined[features].to_numpy(dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any() or not np.equal(x, np.floor(x)).all():
        raise ValueError('Invalid event counts')
    if (x.sum(axis=1) == 0).any():
        raise ValueError('Empty event trace')
    return joined, features

def split_indices(y, seed=SEED):
    ix = np.arange(len(y))
    train, other = train_test_split(ix, test_size=.4, random_state=seed, stratify=y)
    valid, test = train_test_split(other, test_size=.5, random_state=seed, stratify=np.asarray(y)[other])
    return {'train': np.sort(train), 'validation': np.sort(valid), 'test': np.sort(test)}

def metrics(y, scores, threshold):
    y, scores = np.asarray(y), np.asarray(scores)
    if not np.isfinite(scores).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid scoring input')
    tn, fp, fn, tp = confusion_matrix(y, scores >= threshold, labels=[0, 1]).ravel().tolist()
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    fpr = fp / (fp + tn) if fp + tn else np.nan
    p, r, _ = precision_recall_curve(y, scores)
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=f1, fpr=fpr, pr_auc=auc(r, p) if np.count_nonzero(y) else np.nan,
                average_precision=average_precision_score(y, scores) if np.count_nonzero(y) else np.nan)

def choose_thresholds(y, scores, fpr_limit=.01):
    """Evaluate complete score-tie groups. Ties: fewer false alarms, then higher threshold."""
    y, scores = np.asarray(y), np.asarray(scores)
    if not np.isfinite(scores).all() or len(np.unique(y)) != 2:
        raise ValueError('Threshold selection requires finite scores and both classes')
    order = np.argsort(-scores, kind='stable')
    sy, ss = y[order], scores[order]
    end = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(ss) - 1]
    tp = np.r_[0, np.cumsum(sy)[end]]
    fp = np.r_[0, (end + 1) - np.cumsum(sy)[end]]
    thresholds = np.r_[np.nextafter(ss[0], np.inf), ss[end]]
    positives, negatives = y.sum(), len(y) - y.sum()
    f1 = 2 * tp / (positives + tp + fp)
    candidates = np.flatnonzero(f1 == f1.max())
    best = min(candidates, key=lambda i: (fp[i], -thresholds[i]))
    eligible = np.flatnonzero(fp / negatives <= fpr_limit)
    constrained = min(eligible, key=lambda i: (-tp[i], fp[i], -thresholds[i]))
    return {'f1': float(thresholds[best]), 'fpr01': float(thresholds[constrained])}

class NormalRule:
    """Largest absolute deviation from a normal-training marginal mean, scale floor 1."""
    def fit(self, x, y):
        normal = x[np.asarray(y) == 0].astype(float)
        self.mean_ = normal.mean(axis=0)
        self.scale_ = np.maximum(normal.std(axis=0), 1.)
        return self
    def score(self, x):
        return np.max(np.abs((x - self.mean_) / self.scale_), axis=1)
