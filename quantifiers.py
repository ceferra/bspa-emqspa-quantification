"""
Custom QuaPy quantifiers for the quantification benchmark.

Both extend BaseQuantifier with the standard fit(X, y) / predict(X) interface
and are compatible with Ensemble, GridSearchQ, and any QuaPy evaluation protocol.

References: Ferri et al. (2025)
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier


class BSPAQuantifier(BaseQuantifier):
    """
    Beta-Smoothed Scaled Probability Average (Ferri et al., 2025).

    Fits a probabilistic classifier, computes the Scaled Probability Average
    (SPA) on the test set, then shrinks toward the training prior:

        lambda  = n_test / (n_test + kappa)
        pi_hat  = lambda * SPA(test) + (1 - lambda) * pi_train

    This reduces variance when the test set is small; as n_test grows,
    lambda -> 1 and the estimate converges to plain SPA.

    Parameters
    ----------
    classifier : sklearn estimator (default LogisticRegression)
    val_split  : float — fraction held out for SPA calibration
    kappa      : shrinkage strength (default 5)
    """

    def __init__(self, classifier=None, val_split=0.4, kappa=5.0):
        self.classifier = classifier
        self.val_split = val_split
        self.kappa = kappa

    def fit(self, X, y):
        clf = (
            clone(self.classifier)
            if self.classifier is not None
            else LogisticRegression(max_iter=1000, random_state=0)
        )
        lc = LabelledCollection(X, y)
        train_lc, val_lc = lc.split_stratified(
            1.0 - self.val_split, random_state=0
        )
        X_tr, y_tr = train_lc.Xy
        X_val, y_val = val_lc.Xy
        self.clf_ = clf.fit(X_tr, y_tr)
        vp = self.clf_.predict_proba(X_val)[:, 1]
        pos = y_val == 1
        neg = y_val == 0
        self.mu_pos_ = float(vp[pos].mean()) if pos.sum() > 0 else 1.0
        self.mu_neg_ = float(vp[neg].mean()) if neg.sum() > 0 else 0.0
        self.pi_train_ = float(y_val.mean())
        return self

    def predict(self, X):
        probs = self.clf_.predict_proba(X)[:, 1]
        n = len(probs)
        lam = n / (n + self.kappa)
        pa = float(probs.mean())
        d = self.mu_pos_ - self.mu_neg_
        spa = (
            float(np.clip((pa - self.mu_neg_) / d, 0, 1))
            if abs(d) > 1e-9
            else float(np.clip(pa, 0, 1))
        )
        pi_pos = float(np.clip(lam * spa + (1 - lam) * self.pi_train_, 0, 1))
        return np.array([1.0 - pi_pos, pi_pos])

    def get_params(self, deep=True):
        return {
            "classifier": self.classifier,
            "val_split": self.val_split,
            "kappa": self.kappa,
        }

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


class EMQSPAQuantifier(BaseQuantifier):
    """
    EMQ warm-started from SPA (Ferri et al., 2025).

    Runs the Saerens et al. (2002) EM quantifier initialised from the
    SPA estimate instead of the plain probability average.
    The better starting point reduces iteration count and avoids convergence
    to spurious fixed points under strong prior shift.

    Parameters
    ----------
    classifier : sklearn estimator (default LogisticRegression)
    val_split  : float — fraction held out for SPA calibration
    max_iter, tol : EM convergence parameters
    """

    def __init__(self, classifier=None, val_split=0.4, max_iter=1000, tol=1e-6):
        self.classifier = classifier
        self.val_split = val_split
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        clf = (
            clone(self.classifier)
            if self.classifier is not None
            else LogisticRegression(max_iter=1000, random_state=0)
        )
        lc = LabelledCollection(X, y)
        train_lc, val_lc = lc.split_stratified(
            1.0 - self.val_split, random_state=0
        )
        X_tr, y_tr = train_lc.Xy
        X_val, y_val = val_lc.Xy
        self.clf_ = clf.fit(X_tr, y_tr)
        vp = self.clf_.predict_proba(X_val)[:, 1]
        pos = y_val == 1
        neg = y_val == 0
        self.mu_pos_ = float(vp[pos].mean()) if pos.sum() > 0 else 1.0
        self.mu_neg_ = float(vp[neg].mean()) if neg.sum() > 0 else 0.0
        self.pi_train_ = float(y_val.mean())
        return self

    def predict(self, X):
        probs = self.clf_.predict_proba(X)[:, 1]
        pi_tr = self.pi_train_
        if pi_tr < 1e-9 or pi_tr > 1.0 - 1e-9:
            pi_pos = float(probs.mean())
        else:
            pa = float(probs.mean())
            d = self.mu_pos_ - self.mu_neg_
            pi = (
                float(np.clip((pa - self.mu_neg_) / d, 0, 1))
                if abs(d) > 1e-9
                else float(np.clip(pa, 0, 1))
            )
            for _ in range(self.max_iter):
                num = (pi / pi_tr) * probs
                den = num + ((1 - pi) / (1 - pi_tr)) * (1 - probs)
                den = np.where(den < 1e-12, 1e-12, den)
                pi_new = float((num / den).mean())
                if abs(pi_new - pi) < self.tol:
                    break
                pi = pi_new
            pi_pos = float(np.clip(pi, 0, 1))
        return np.array([1.0 - pi_pos, pi_pos])

    def get_params(self, deep=True):
        return {
            "classifier": self.classifier,
            "val_split": self.val_split,
            "max_iter": self.max_iter,
            "tol": self.tol,
        }

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


class PACCFixed(BaseQuantifier):
    def __init__(self, classifier=None, val_split=0.4):
        self.classifier = classifier
        self.val_split = val_split

    def fit(self, X, y):
        clf = (
            clone(self.classifier)
            if self.classifier is not None
            else LogisticRegression(max_iter=1000, random_state=0)
        )
        lc = LabelledCollection(X, y)
        train_lc, val_lc = lc.split_stratified(
            1.0 - self.val_split, random_state=0
        )
        X_tr, y_tr = train_lc.Xy
        X_val, y_val = val_lc.Xy
        self.clf_ = clf.fit(X_tr, y_tr)
        vp = self.clf_.predict_proba(X_val)[:, 1]
        pos = y_val == 1
        neg = y_val == 0
        self.mu_pos_ = float(vp[pos].mean()) if pos.sum() > 0 else 1.0
        self.mu_neg_ = float(vp[neg].mean()) if neg.sum() > 0 else 0.0
        return self

    def predict(self, X):
        probs = self.clf_.predict_proba(X)[:, 1]
        pa = float(probs.mean())
        d = self.mu_pos_ - self.mu_neg_
        if abs(d) > 1e-9:
            pi_pos = float(np.clip((pa - self.mu_neg_) / d, 0, 1))
        else:
            pi_pos = float(np.clip(pa, 0, 1))
        return np.array([1.0 - pi_pos, pi_pos])

    def get_params(self, deep=True):
        return {"classifier": self.classifier, "val_split": self.val_split}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self
