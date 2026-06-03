"""
Core benchmark: evaluate all quantification methods on UCI datasets using APP.

Usage
-----
    python benchmark.py [--out results.pkl] [--splits 5] [--jobs 1]

Results are saved as a pickle dict:
    results[dataset_name][method_name] = {
        'mse': [float, ...],
        'mae': [float, ...],
        'fit_time':     [float, ...],   # seconds to fit on training set
        'predict_time': [float, ...],   # seconds per quantify() call (avg over APP samples)
    }
"""

import argparse
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import quapy as qp
import quapy.evaluation as qpe
from quapy.protocol import APP
from tqdm import tqdm

from datasets import load_datasets
from methods import make_methods, method_names

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Defaults (override via CLI or config)
# --------------------------------------------------------------------------- #
DEFAULT_OUT        = "results.pkl"
N_SPLITS           = 5
N_PREVALENCES      = 21   # 0 %, 5 %, …, 100 %
APP_REPEATS        = 10   # samples per prevalence level → 210 total
SAMPLE_SIZE        = 100  # APP sample size


def run_benchmark(datasets, n_splits=N_SPLITS, n_prevalences=N_PREVALENCES,
                  app_repeats=APP_REPEATS, sample_size=SAMPLE_SIZE):
    """
    Run the full APP benchmark.

    Parameters
    ----------
    datasets : dict  {name: LabelledCollection}
    n_splits, n_prevalences, app_repeats, sample_size : int

    Returns
    -------
    results : dict
        results[ds_name][method_name] = {'mse': [float], 'mae': [float]}
    active : list of str
        Methods that produced at least one valid result.
    """
    qp.environ["SAMPLE_SIZE"] = sample_size
    np.random.seed(42)

    all_method_names = method_names()
    n_app_samples = n_prevalences * app_repeats
    results = {}

    for ds_name, lc_full in tqdm(datasets.items(), desc="Datasets"):
        ds_res = {
            m: {"mse": [], "mae": [], "fit_time": [], "predict_time": []}
            for m in all_method_names
        }

        for seed in range(n_splits):
            train_lc, test_lc = lc_full.split_stratified(
                train_prop=0.70, random_state=seed
            )
            if len(np.unique(train_lc.labels)) < 2:
                continue
            if len(np.unique(test_lc.labels)) < 2:
                continue

            protocol = APP(
                test_lc,
                n_prevalences=n_prevalences,
                repeats=app_repeats,
                random_state=seed,
            )

            X_tr, y_tr = train_lc.Xy
            methods = make_methods()

            for m_name, q in methods.items():
                try:
                    # ── fit timing ──────────────────────────────────────────
                    t0 = time.perf_counter()
                    q.fit(X_tr, y_tr)
                    fit_time = time.perf_counter() - t0

                    # ── evaluation + predict timing ─────────────────────────
                    t1 = time.perf_counter()
                    mse_v = qpe.evaluate(q, protocol=protocol, error_metric="mse")
                    predict_time = (time.perf_counter() - t1) / n_app_samples

                    mae_v = qpe.evaluate(q, protocol=protocol, error_metric="mae")

                    ds_res[m_name]["mse"].append(float(mse_v))
                    ds_res[m_name]["mae"].append(float(mae_v))
                    ds_res[m_name]["fit_time"].append(fit_time)
                    ds_res[m_name]["predict_time"].append(predict_time)
                except Exception:
                    pass   # skip silently; method failed on this split

        results[ds_name] = ds_res

    active = [
        m for m in all_method_names
        if any(results[ds][m]["mse"] for ds in results)
    ]
    print(f"\nActive methods ({len(active)}): {active}")
    return results, active


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="QuaPy quantification benchmark")
    p.add_argument("--out",         default=DEFAULT_OUT,   help="Output pickle file")
    p.add_argument("--splits",      type=int, default=N_SPLITS)
    p.add_argument("--prevalences", type=int, default=N_PREVALENCES)
    p.add_argument("--repeats",     type=int, default=APP_REPEATS)
    p.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading datasets …")
    datasets, skipped = load_datasets()

    print(f"\nStarting benchmark — {len(datasets)} datasets, "
          f"{args.splits} splits, APP(n_prev={args.prevalences}, "
          f"repeats={args.repeats}, sample_size={args.sample_size})")

    results, active = run_benchmark(
        datasets,
        n_splits=args.splits,
        n_prevalences=args.prevalences,
        app_repeats=args.repeats,
        sample_size=args.sample_size,
    )

    out_path = Path(args.out)
    with open(out_path, "wb") as fh:
        pickle.dump({"results": results, "active": active,
                     "datasets_info": {k: len(v) for k, v in datasets.items()}},
                    fh)
    print(f"\nResults saved to {out_path}")
