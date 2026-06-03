"""
Plotting and statistical analysis for the quantification benchmark.

Functions
---------
plot_mse_bar         — bar chart of average MSE per method
plot_cd_diagram      — Nemenyi critical difference diagram
plot_app_curve       — MSE vs. test prevalence curve (APP)
plot_diagonal        — predicted vs. true prevalence scatter plots
results_table        — styled pandas DataFrame of per-dataset MSE
friedman_test        — Friedman + average-rank analysis
"""

import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare
from tqdm import tqdm

import quapy as qp
from quapy.protocol import APP

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Tag sets
# --------------------------------------------------------------------------- #
OUR_METHODS   = {"BSPA", "EMQSPA"}
NEWQP_METHODS = {"KDEy-HD", "KDEy-ML", "Ens-PACC"}

# Per-method line styles for the APP curve
STYLES = {
    "CC":        dict(color="#bdc3c7", ls="-",  lw=0.9),
    "PCC":       dict(color="#3498db", ls="--", lw=0.9),
    "ACC":       dict(color="#e74c3c", ls="--", lw=1.2),
    "PACC":      dict(color="#8e44ad", ls="-.", lw=1.2),
    "EMQ":       dict(color="#95a5a6", ls="--", lw=1.2),
    "HDy":       dict(color="#aed6f1", ls=":",  lw=1.5),
    "MS":        dict(color="#d35400", ls="--", lw=1.5),
    "KDEy-HD":   dict(color="#2980b9", ls="-",  lw=2.0),
    "KDEy-ML":   dict(color="#1a5276", ls="-",  lw=2.2),
    "Ens-PACC":  dict(color="#6c3483", ls="-.", lw=2.0),
    "BSPA":      dict(color="#e67e22", ls="-",  lw=2.5),
    "EMQSPA":    dict(color="#1abc9c", ls="-",  lw=2.0),
}


def _method_color(m):
    if m in OUR_METHODS:   return "#e67e22"
    if m in NEWQP_METHODS: return "#2980b9"
    if m in {"CC", "PCC"}: return "#bdc3c7"
    return "#27ae60"


def _avg_metric(results, ds, m, key):
    v = results[ds][m][key]
    return np.mean(v) if v else np.nan


# --------------------------------------------------------------------------- #
# Results table
# --------------------------------------------------------------------------- #

def results_table(results, datasets, active):
    """Return a styled pandas DataFrame of per-dataset average MSE."""
    rows = []
    for ds in results:
        row = {
            "Dataset": ds,
            "N":    len(datasets[ds]),
            "pi(+)": f"{datasets[ds].prevalence()[1]:.2f}",
        }
        for m in active:
            row[m] = _avg_metric(results, ds, m, "mse")
        rows.append(row)

    avg_row = {"Dataset": "AVERAGE", "N": "", "pi(+)": ""}
    for m in active:
        avg_row[m] = np.nanmean([r[m] for r in rows])
    rows.append(avg_row)
    df = pd.DataFrame(rows)

    def _hl(row):
        if row["Dataset"] == "AVERAGE":
            return [""] * len(row)
        v = row[active]
        if v.isna().all():
            return [""] * len(row)
        mv = v.min()
        out = []
        for col in row.index:
            if col in active and row[col] == mv:
                c = ("#aed6f1" if col in NEWQP_METHODS
                     else "#a9dfbf" if col in OUR_METHODS
                     else "#f9e79f")
                out.append(f"background-color:{c}")
            else:
                out.append("")
        return out

    styled = (
        df.style
        .format({m: "{:.5f}" for m in active})
        .apply(_hl, axis=1)
        .set_caption(
            "MSE (APP, UCI datasets). "
            "Blue=QuaPy new, green=our proposals, yellow=existing best."
        )
    )
    return styled, df


# --------------------------------------------------------------------------- #
# Average MSE per method
# --------------------------------------------------------------------------- #

def avg_mse_per_method(results, active):
    """Return {method: mean_mse} averaged across all datasets."""
    return {
        m: np.nanmean([_avg_metric(results, ds, m, "mse") for ds in results])
        for m in active
    }


def avg_time_per_method(results, active, key="fit_time"):
    """Return {method: mean_time} averaged across datasets and splits."""
    return {
        m: np.nanmean([_avg_metric(results, ds, m, key) for ds in results])
        for m in active
    }


# --------------------------------------------------------------------------- #
# Time bar chart (fit + predict)
# --------------------------------------------------------------------------- #

def plot_time_bars(avg_fit, avg_pred, save_path=None):
    """Two horizontal bar charts: fit time (left) and predict time per sample (right)."""
    methods = sorted(avg_fit, key=avg_fit.get, reverse=True)
    colors  = [_method_color(m) for m in methods]
    y_pos   = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(13, max(5, len(methods) * 0.55)))

    for ax, data, title, unit in [
        (axes[0], avg_fit,  "Training time (fit)",           "seconds"),
        (axes[1], avg_pred, "Inference time per sample (predict)", "seconds"),
    ]:
        vals = [data[m] for m in methods]
        bars = ax.barh(y_pos, vals, color=colors, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.4f}s", va="center", fontsize=7.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(methods, fontsize=9)
        ax.set_xlabel(unit)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Computational cost per method  (orange=proposals, blue=QuaPy new, green=baselines)",
                 fontsize=9, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
# Pareto scatter: MSE vs. time
# --------------------------------------------------------------------------- #

def plot_pareto_mse_time(avg_mse, avg_time, time_label="Training time (s)",
                         save_path=None):
    """
    Scatter plot of average MSE vs. average time.
    The Pareto-optimal frontier (lower MSE *and* lower time) is drawn as a step line.
    """
    methods = list(avg_mse.keys())
    x = np.array([avg_time[m] for m in methods])
    y = np.array([avg_mse[m]  for m in methods])

    # ── compute Pareto frontier ──────────────────────────────────────────────
    order = np.argsort(x)
    pareto_idx = []
    best_mse = np.inf
    for i in order:
        if y[i] < best_mse:
            best_mse = y[i]
            pareto_idx.append(i)

    pareto_x = x[pareto_idx]
    pareto_y = y[pareto_idx]
    # extend step line to the right
    step_x = np.concatenate([[pareto_x[0]], np.repeat(pareto_x[1:], 2),
                              [x.max() * 1.05]])
    step_y = np.concatenate([np.repeat(pareto_y[:-1], 2), [pareto_y[-1], pareto_y[-1]]])

    fig, ax = plt.subplots(figsize=(10, 6))

    # Pareto step line
    ax.plot(step_x, step_y, color="#aaaaaa", lw=1.2, ls="--",
            label="Pareto frontier", zorder=1)

    # Scatter points
    for m, xi, yi in zip(methods, x, y):
        c    = _method_color(m)
        mark = "*" if m in OUR_METHODS else "o"
        size = 160 if m in OUR_METHODS else 90
        ax.scatter(xi, yi, color=c, marker=mark, s=size, zorder=5,
                   edgecolors="white" if m not in OUR_METHODS else "black",
                   linewidths=0.5)
        # label offset to avoid overlaps
        dx = (x.max() - x.min()) * 0.01
        dy = (y.max() - y.min()) * 0.008
        ax.annotate(m, (xi, yi), xytext=(xi + dx, yi + dy),
                    fontsize=8, color=c, fontweight="bold")

    ax.set_xlabel(time_label, fontsize=10)
    ax.set_ylabel("Average MSE  (APP, 30 datasets)", fontsize=10)
    ax.set_title("Accuracy–efficiency trade-off\n"
                 "★ = our proposals  |  orange=proposals  |  "
                 "blue=QuaPy new  |  green=baselines",
                 fontsize=9)

    # Custom legend patches
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#e67e22",
               markersize=13, label="Our proposals"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2980b9",
               markersize=9,  label="QuaPy new"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60",
               markersize=9,  label="QuaPy baselines"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bdc3c7",
               markersize=9,  label="Classical baselines"),
        Line2D([0], [0], ls="--", color="#aaaaaa", lw=1.2, label="Pareto frontier"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# --------------------------------------------------------------------------- #
# Bar chart
# --------------------------------------------------------------------------- #

def plot_mse_bar(avg_mse_dict, save_path=None):
    sorted_m = sorted(avg_mse_dict, key=avg_mse_dict.get)
    fig, ax = plt.subplots(figsize=(max(10, len(sorted_m)), 4))
    bars = ax.bar(
        sorted_m,
        [avg_mse_dict[m] for m in sorted_m],
        color=[_method_color(m) for m in sorted_m],
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, m in zip(bars, sorted_m):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.00003,
            f"{avg_mse_dict[m]:.5f}",
            ha="center", va="bottom", fontsize=7.5,
        )
    ax.set_ylabel("Average MSE  (APP, all datasets)")
    ax.set_title("orange=our proposals  |  blue=QuaPy new  |  green=QuaPy baselines")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# --------------------------------------------------------------------------- #
# Friedman + CD diagram
# --------------------------------------------------------------------------- #

def friedman_test(results, active):
    """
    Compute Friedman test and average ranks.

    Returns
    -------
    rank_df  : DataFrame  (Method, Avg Rank)
    stat, pval, CD : floats
    mse_clean, N, K : array and dimensions
    """
    mse_matrix = np.array([
        [_avg_metric(results, ds, m, "mse") for m in active]
        for ds in results
    ])
    valid_rows = ~np.isnan(mse_matrix).any(axis=1)
    mse_clean  = mse_matrix[valid_rows]
    N, K = mse_clean.shape
    print(f"Complete matrix: {N} datasets × {K} methods")

    stat, pval = friedmanchisquare(*[mse_clean[:, j] for j in range(K)])
    print(f"Friedman chi²={stat:.3f}  p={pval:.3e}")

    ranks    = np.array([stats.rankdata(row) for row in mse_clean])
    avg_rnks = ranks.mean(axis=0)
    rank_df  = (
        pd.DataFrame({"Method": active, "Avg Rank": avg_rnks})
        .sort_values("Avg Rank")
        .reset_index(drop=True)
    )
    rank_df.index += 1

    q_tbl = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850,
        7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219,
        12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
    }
    q_alpha = q_tbl.get(K, 3.5)
    CD = q_alpha * np.sqrt(K * (K + 1) / (6 * N))
    print(f"CD (α=0.05, k={K}, N={N}) = {CD:.3f}")

    return rank_df, stat, pval, CD, mse_clean, N, K


def plot_cd_diagram(rank_df, CD, N, save_path=None):
    sorted_names  = rank_df["Method"].tolist()
    sorted_ranks  = rank_df["Avg Rank"].tolist()
    K = len(sorted_names)

    fig, ax = plt.subplots(figsize=(max(10, K + 3), 3.8))
    ax.set_xlim(0.5, K + 0.5)
    ax.set_ylim(-1.9, 2.3)
    ax.set_yticks([])
    ax.set_xlabel("Average Rank")
    ax.set_title(f"Nemenyi CD Diagram  (CD={CD:.2f}, α=0.05, N={N})", fontsize=10)
    ax.hlines(0, 0.5, K + 0.5, colors="#bdc3c7", lw=0.7)

    for i, (m, r) in enumerate(zip(sorted_names, sorted_ranks)):
        c = _method_color(m)
        offset = 0.68 if i % 2 == 0 else -0.68
        ax.plot(r, 0, "o", color=c, ms=10, zorder=5)
        ax.annotate(
            m, (r, 0), xytext=(r, offset), ha="center",
            fontsize=8, color=c, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=c, lw=0.8),
        )

    best = sorted_ranks[0]
    ax.annotate(
        "", xy=(best + CD, 1.6), xytext=(best, 1.6),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.4),
    )
    ax.text(best + CD / 2, 1.78, f"CD={CD:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# --------------------------------------------------------------------------- #
# APP curve (MSE vs. prevalence)
# --------------------------------------------------------------------------- #

def compute_app_curve(datasets, focus_methods, n_prevalences=21, repeats=3):
    """
    Re-evaluate selected methods per prevalence level.

    Returns
    -------
    prev_errors : dict  {method: {prevalence: [mse, ...]}}
    """
    prev_errors = {m: defaultdict(list) for m in focus_methods}

    for ds_name, lc_full in tqdm(datasets.items(), desc="APP curve"):
        train_lc, test_lc = lc_full.split_stratified(0.70, random_state=0)
        if len(np.unique(train_lc.labels)) < 2:
            continue
        if len(np.unique(test_lc.labels)) < 2:
            continue

        from methods import make_methods
        protocol = APP(test_lc, n_prevalences=n_prevalences,
                       repeats=repeats, random_state=0)
        samples = list(protocol())

        for m_name in focus_methods:
            q = make_methods()[m_name]
            try:
                q.fit(*train_lc.Xy)
            except Exception:
                continue
            for X_s, tp_vec in samples:
                try:
                    est = q.predict(X_s)
                    tp  = round(float(tp_vec[1]), 2)
                    mse = (tp - float(est[1])) ** 2
                    prev_errors[m_name][tp].append(mse)
                except Exception:
                    pass

    return prev_errors


def plot_app_curve(prev_errors, focus_methods, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in focus_methods:
        pts = sorted(prev_errors[m].keys())
        if not pts:
            continue
        y = [np.mean(prev_errors[m][p]) for p in pts]
        ax.plot([p * 100 for p in pts], y, label=m,
                **STYLES.get(m, dict(lw=1.0)))
    ax.set_xlabel("Test positive prevalence (%)")
    ax.set_ylabel("Mean MSE (averaged over datasets)")
    ax.set_title("MSE vs. test prevalence (APP, mean over all datasets)")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# --------------------------------------------------------------------------- #
# Diagonal plot
# --------------------------------------------------------------------------- #

def plot_diagonal(datasets, focus_methods, active,
                  diag_ds=None, n_prevalences=21, repeats=5,
                  save_path=None):
    if diag_ds is None:
        diag_ds = "german" if "german" in datasets else list(datasets.keys())[0]

    focus_methods = [m for m in focus_methods if m in active]
    lc_d = datasets[diag_ds]
    tr_d, te_d = lc_d.split_stratified(0.70, random_state=0)
    prot_d = APP(te_d, n_prevalences=n_prevalences, repeats=repeats, random_state=0)
    samples_d = list(prot_d())

    from methods import make_methods
    fig, axes = plt.subplots(1, len(focus_methods), figsize=(4 * len(focus_methods), 3.8))
    if len(focus_methods) == 1:
        axes = [axes]

    for ax, m_name in zip(axes, focus_methods):
        q = make_methods()[m_name]
        try:
            q.fit(*tr_d.Xy)
        except Exception:
            ax.set_title(f"{m_name} (failed)")
            continue

        true_p, est_p = [], []
        for X_s, tp_vec in samples_d:
            try:
                est = q.predict(X_s)
                true_p.append(float(tp_vec[1]))
                est_p.append(float(est[1]))
            except Exception:
                pass

        c = _method_color(m_name)
        ax.scatter(true_p, est_p, alpha=0.35, s=12, color=c)
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        mae_v = np.mean(np.abs(np.array(true_p) - np.array(est_p))) if true_p else float("nan")
        ax.set_title(f"{m_name}\nMAE={mae_v:.4f}", fontsize=9)
        ax.set_xlabel("True prevalence")
        if ax is axes[0]:
            ax.set_ylabel("Estimated prevalence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle(f"Diagonal plot — {diag_ds} (APP)", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
