"""
Ablation 3: EM iteration count — EMQ vs EMQSPA.
Both methods converge to the same fixed point; EMQSPA (warm-started from SPA)
should require fewer iterations.
Saves results to ablation_convergence.pkl.
"""
import pickle, warnings
from pathlib import Path
import numpy as np
import quapy as qp
from quapy.protocol import APP
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from sklearn.base import clone
from datasets import load_datasets

warnings.filterwarnings("ignore")

N_SPLITS    = 5
N_PREV      = 21
APP_REPEATS = 10
SAMPLE_SIZE = 100
VAL_SPLIT   = 0.4
MAX_ITER    = 1000
TOL         = 1e-6

def make_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=0))])


class EMQCounting(BaseQuantifier):
    """EMQ initialised from PA, counts iterations."""
    def __init__(self, classifier=None, val_split=0.4):
        self.classifier = classifier
        self.val_split = val_split

    def fit(self, X, y):
        clf = clone(self.classifier) if self.classifier else LogisticRegression(max_iter=1000)
        lc = LabelledCollection(X, y)
        tr, val = lc.split_stratified(1 - self.val_split, random_state=0)
        self.clf_ = clf.fit(*tr.Xy)
        self.pi_train_ = float(val.Xy[1].mean())
        return self

    def predict(self, X):
        probs = self.clf_.predict_proba(X)[:, 1]
        pi_tr = self.pi_train_
        pi = float(probs.mean())   # PA initialisation
        iters = 0
        for i in range(MAX_ITER):
            iters += 1
            num = (pi / pi_tr) * probs
            den = num + ((1 - pi) / (1 - pi_tr)) * (1 - probs)
            den = np.where(den < 1e-12, 1e-12, den)
            pi_new = float((num / den).mean())
            if abs(pi_new - pi) < TOL:
                break
            pi = pi_new
        self._iters = iters
        return np.array([1 - pi, pi])

    def get_params(self, deep=True): return {}
    def set_params(self, **p): return self


class EMQSPACounting(BaseQuantifier):
    """EMQSPA (warm-started from SPA), counts iterations."""
    def __init__(self, classifier=None, val_split=0.4):
        self.classifier = classifier
        self.val_split = val_split

    def fit(self, X, y):
        clf = clone(self.classifier) if self.classifier else LogisticRegression(max_iter=1000)
        lc = LabelledCollection(X, y)
        tr, val = lc.split_stratified(1 - self.val_split, random_state=0)
        X_v, y_v = val.Xy
        self.clf_ = clf.fit(*tr.Xy)
        vp = self.clf_.predict_proba(X_v)[:, 1]
        pos, neg = y_v == 1, y_v == 0
        self.mu_pos_  = float(vp[pos].mean()) if pos.sum() > 0 else 1.0
        self.mu_neg_  = float(vp[neg].mean()) if neg.sum() > 0 else 0.0
        self.pi_train_ = float(y_v.mean())
        return self

    def predict(self, X):
        probs = self.clf_.predict_proba(X)[:, 1]
        pi_tr = self.pi_train_
        pa = float(probs.mean())
        d = self.mu_pos_ - self.mu_neg_
        pi = float(np.clip((pa - self.mu_neg_) / d, 0, 1)) if abs(d) > 1e-9 else float(np.clip(pa, 0, 1))
        iters = 0
        for i in range(MAX_ITER):
            iters += 1
            num = (pi / pi_tr) * probs
            den = num + ((1 - pi) / (1 - pi_tr)) * (1 - probs)
            den = np.where(den < 1e-12, 1e-12, den)
            pi_new = float((num / den).mean())
            if abs(pi_new - pi) < TOL:
                break
            pi = pi_new
        self._iters = iters
        return np.array([1 - pi, pi])

    def get_params(self, deep=True): return {}
    def set_params(self, **p): return self


def run():
    qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
    np.random.seed(42)
    print("Loading datasets …")
    datasets, _ = load_datasets()

    emq_iters  = []
    spa_iters  = []

    for ds_name, lc_full in tqdm(datasets.items(), desc="Datasets"):
        for seed in range(N_SPLITS):
            tr_lc, te_lc = lc_full.split_stratified(0.70, random_state=seed)
            if len(np.unique(tr_lc.labels)) < 2 or len(np.unique(te_lc.labels)) < 2:
                continue
            X_tr, y_tr = tr_lc.Xy
            protocol = APP(te_lc, n_prevalences=N_PREV, repeats=APP_REPEATS, random_state=seed)
            samples = list(protocol())

            for Cls, store in [(EMQCounting, emq_iters), (EMQSPACounting, spa_iters)]:
                q = Cls(make_clf(), val_split=VAL_SPLIT)
                try:
                    q.fit(X_tr, y_tr)
                    for X_s, _ in samples:
                        q.predict(X_s)
                        store.append(q._iters)
                except Exception:
                    pass

    results = {"emq_iters": emq_iters, "emqspa_iters": spa_iters}
    Path("ablation_convergence.pkl").write_bytes(pickle.dumps(results))
    print("Saved ablation_convergence.pkl")
    print(f"\nEMQ    mean iters: {np.mean(emq_iters):.2f}  (median {np.median(emq_iters):.0f})")
    print(f"EMQSPA mean iters: {np.mean(spa_iters):.2f}  (median {np.median(spa_iters):.0f})")

if __name__ == "__main__":
    run()
