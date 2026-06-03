"""
Ablation 2: sensitivity of BSPA, PACC, EMQ, KDEy-HD to test sample size.
Runs APP with sample_size in SIZES on 30 UCI datasets.
Saves results to ablation_samplesize.pkl.
"""
import pickle, warnings
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
from quapy.method.aggregative import KDEyHD
from datasets import load_datasets
from quantifiers import BSPAQuantifier, PACCFixed

warnings.filterwarnings("ignore")

SIZES      = [25, 50, 100, 200, 500]
N_SPLITS   = 5
N_PREV     = 21
APP_REPEATS= 10
VAL_SPLIT  = 0.4
METHODS    = ["BSPA", "PACC", "EMQ", "KDEy-HD"]

def make_clf():
    return Pipeline([("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=0))])

def run():
    np.random.seed(42)
    print("Loading datasets …")
    datasets, _ = load_datasets()

    # results[sample_size][method][ds_name] = [mse per split]
    results = {sz: {m: {} for m in METHODS} for sz in SIZES}

    for ds_name, lc_full in tqdm(datasets.items(), desc="Datasets"):
        for sz in SIZES:
            for m in METHODS:
                results[sz][m][ds_name] = []

        for seed in range(N_SPLITS):
            tr_lc, te_lc = lc_full.split_stratified(0.70, random_state=seed)
            if len(np.unique(tr_lc.labels)) < 2 or len(np.unique(te_lc.labels)) < 2:
                continue
            X_tr, y_tr = tr_lc.Xy

            # Fit once per split (shared across sample sizes)
            quantifiers = {
                "BSPA":    BSPAQuantifier(make_clf(), val_split=VAL_SPLIT, kappa=5.0),
                "PACC":    qpa.PACC(make_clf(), val_split=VAL_SPLIT),
                "EMQ":     qpa.EMQ(make_clf()),
                "KDEy-HD": KDEyHD(make_clf()),
            }
            fitted = {}
            for name, q in quantifiers.items():
                try:
                    q.fit(X_tr, y_tr)
                    fitted[name] = q
                except Exception: pass

            for sz in SIZES:
                qp.environ["SAMPLE_SIZE"] = sz
                protocol = APP(te_lc, n_prevalences=N_PREV, repeats=APP_REPEATS,
                               random_state=seed)
                for name, q in fitted.items():
                    try:
                        mse = float(qpe.evaluate(q, protocol=protocol, error_metric="mse"))
                        results[sz][name][ds_name].append(mse)
                    except Exception: pass

    Path("ablation_samplesize.pkl").write_bytes(pickle.dumps(results))
    print("Saved ablation_samplesize.pkl")

    print("\nAvg MSE by sample size:")
    print(f"{'Size':>6}  " + "  ".join(f"{m:>10}" for m in METHODS))
    for sz in SIZES:
        row = f"{sz:>6}  "
        for m in METHODS:
            avg = np.nanmean([np.mean(v) for v in results[sz][m].values() if v])
            row += f"  {avg:>10.5f}"
        print(row)

if __name__ == "__main__":
    run()
