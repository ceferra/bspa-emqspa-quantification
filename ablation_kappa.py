"""
Ablation 1: sensitivity of BSPA to the shrinkage parameter kappa.
Runs the full QuaPy APP benchmark for BSPA with kappa in KAPPA_GRID,
plus PACC and EMQ as fixed baselines, on 30 UCI datasets.
Saves results to ablation_kappa.pkl.
"""
import pickle, time, warnings
from pathlib import Path
import numpy as np
import quapy as qp
import quapy.evaluation as qpe
from quapy.protocol import APP
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import quapy.method.aggregative as qpa
from datasets import load_datasets
from quantifiers import BSPAQuantifier

warnings.filterwarnings("ignore")

KAPPA_GRID   = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
N_SPLITS     = 5
N_PREV       = 21
APP_REPEATS  = 10
SAMPLE_SIZE  = 100
VAL_SPLIT    = 0.4

def make_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=0))])

def run():
    qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE
    np.random.seed(42)
    print("Loading datasets …")
    datasets, _ = load_datasets()
    n_samples = N_PREV * APP_REPEATS

    # keys: method_name -> {ds_name: [mse per split]}
    results = {f"BSPA_k{k}": {} for k in KAPPA_GRID}
    results["PACC"] = {}
    results["EMQ"]  = {}

    for ds_name, lc_full in tqdm(datasets.items(), desc="Datasets"):
        for key in results:
            results[key][ds_name] = []

        for seed in range(N_SPLITS):
            tr_lc, te_lc = lc_full.split_stratified(0.70, random_state=seed)
            if len(np.unique(tr_lc.labels)) < 2 or len(np.unique(te_lc.labels)) < 2:
                continue
            protocol = APP(te_lc, n_prevalences=N_PREV, repeats=APP_REPEATS, random_state=seed)
            X_tr, y_tr = tr_lc.Xy

            # PACC
            q = qpa.PACC(make_clf(), val_split=VAL_SPLIT)
            try:
                q.fit(X_tr, y_tr)
                results["PACC"][ds_name].append(
                    float(qpe.evaluate(q, protocol=protocol, error_metric="mse")))
            except Exception: pass

            # EMQ
            q = qpa.EMQ(make_clf())
            try:
                q.fit(X_tr, y_tr)
                results["EMQ"][ds_name].append(
                    float(qpe.evaluate(q, protocol=protocol, error_metric="mse")))
            except Exception: pass

            # BSPA variants
            for k in KAPPA_GRID:
                q = BSPAQuantifier(make_clf(), val_split=VAL_SPLIT, kappa=k)
                try:
                    q.fit(X_tr, y_tr)
                    results[f"BSPA_k{k}"][ds_name].append(
                        float(qpe.evaluate(q, protocol=protocol, error_metric="mse")))
                except Exception: pass

    Path("ablation_kappa.pkl").write_bytes(pickle.dumps(results))
    print("Saved ablation_kappa.pkl")

    # Print summary
    print("\nAvg MSE per kappa:")
    for k in KAPPA_GRID:
        key = f"BSPA_k{k}"
        avg = np.nanmean([np.mean(v) for v in results[key].values() if v])
        print(f"  kappa={k:<5}  MSE={avg:.5f}")
    for ref in ["PACC", "EMQ"]:
        avg = np.nanmean([np.mean(v) for v in results[ref].values() if v])
        print(f"  {ref:<12}  MSE={avg:.5f}")

if __name__ == "__main__":
    run()
