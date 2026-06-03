"""
Analysis script — load benchmark results and produce all figures/tables.

Usage
-----
    python analyse.py [--results results.pkl] [--out-dir figures/]

Generates:
    figures/mse_bar.png
    figures/cd_diagram.png
    figures/app_curve.png
    figures/diagonal.png
"""

import argparse
import pickle
import warnings
from pathlib import Path

import quapy as qp

from datasets import load_datasets
from methods import method_names
from plots import (
    avg_mse_per_method,
    avg_time_per_method,
    compute_app_curve,
    friedman_test,
    plot_app_curve,
    plot_cd_diagram,
    plot_diagonal,
    plot_mse_bar,
    plot_pareto_mse_time,
    plot_time_bars,
    results_table,
)

warnings.filterwarnings("ignore")

FOCUS_CURVE    = ["ACC", "PACC", "EMQ", "HDy", "KDEy-ML", "MS", "BSPA", "EMQSPA"]
FOCUS_DIAGONAL = ["PACC", "EMQ", "KDEy-ML", "BSPA", "EMQSPA"]


def parse_args():
    p = argparse.ArgumentParser(description="Quantification benchmark analysis")
    p.add_argument("--results", default="results.pkl",  help="Benchmark pickle file")
    p.add_argument("--out-dir", default="figures",       help="Output directory for figures")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load results ─────────────────────────────────────────────────────────
    with open(args.results, "rb") as fh:
        data = pickle.load(fh)
    results = data["results"]
    active  = data["active"]
    print(f"Loaded results: {len(results)} datasets, {len(active)} active methods")

    # ── Reload datasets (needed for plots and summary) ────────────────────────
    datasets, _ = load_datasets()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n--- Results table (MSE) ---")
    _, df = results_table(results, datasets, active)
    df.to_csv(out_dir / "mse_table.csv", index=False)
    print(df.tail(3).to_string())

    # ── Average MSE bar chart ─────────────────────────────────────────────────
    avg_mse = avg_mse_per_method(results, active)
    print("\n--- Average MSE ---")
    for m, v in sorted(avg_mse.items(), key=lambda x: x[1]):
        print(f"  {m:<14s}  {v:.5f}")
    plot_mse_bar(avg_mse, save_path=out_dir / "mse_bar.png")

    # ── Friedman + CD diagram ─────────────────────────────────────────────────
    print("\n--- Friedman test ---")
    rank_df, stat, pval, CD, _, N, _ = friedman_test(results, active)
    print(rank_df.to_string())
    plot_cd_diagram(rank_df, CD, N, save_path=out_dir / "cd_diagram.png")

    # ── APP curve ─────────────────────────────────────────────────────────────
    print("\n--- Computing APP curve ---")
    qp.environ["SAMPLE_SIZE"] = 100
    focus_curve = [m for m in FOCUS_CURVE if m in active]
    prev_errors = compute_app_curve(datasets, focus_curve)
    plot_app_curve(prev_errors, focus_curve, save_path=out_dir / "app_curve.png")

    # ── Diagonal plots ────────────────────────────────────────────────────────
    print("\n--- Diagonal plot ---")
    plot_diagonal(datasets, FOCUS_DIAGONAL, active,
                  save_path=out_dir / "diagonal.png")

    # ── Timing analysis ───────────────────────────────────────────────────────
    has_timing = any(
        results[ds][active[0]].get("fit_time")
        for ds in results
    )
    if has_timing:
        print("\n--- Timing analysis ---")
        avg_fit  = avg_time_per_method(results, active, key="fit_time")
        avg_pred = avg_time_per_method(results, active, key="predict_time")

        print("Avg fit time (s):")
        for m, v in sorted(avg_fit.items(), key=lambda x: x[1]):
            print(f"  {m:<14s}  {v:.4f}s")
        print("Avg predict time per sample (s):")
        for m, v in sorted(avg_pred.items(), key=lambda x: x[1]):
            print(f"  {m:<14s}  {v:.6f}s")

        plot_time_bars(avg_fit, avg_pred,
                       save_path=out_dir / "time_bars.png")
        plot_pareto_mse_time(avg_mse, avg_fit,
                             time_label="Average training time (s)",
                             save_path=out_dir / "pareto_mse_fit.png")
        plot_pareto_mse_time(avg_mse, avg_pred,
                             time_label="Average inference time per sample (s)",
                             save_path=out_dir / "pareto_mse_predict.png")
    else:
        print("\n[INFO] No timing data in results — skipping timing plots.")
        print("       Re-run benchmark.py to collect timing information.")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
