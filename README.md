# BSPA: Competitive Quantification at Low Computational Cost

Code and experiments for the paper:

> **BSPA: Competitive Quantification at Low Computational Cost**  
> Cèsar Ferri et al. *Workshop on Learning to Quantify (LQ @ ECML-PKDD 2025)* — **under review**.  
> 📄 [paper.pdf](paper.pdf)

---

## Overview

This repository contains **BSPA** (*Beta-Smoothed Scaled Probability Average*), a lightweight quantifier for binary prevalence estimation under label shift, built on top of [QuaPy](https://github.com/HLT-ISTI/QuaPy), along with all the code needed to reproduce the experiments reported in the paper.

**BSPA** shrinks the SPA estimate toward the training prior with a single hyperparameter κ:

```
λ = n / (n + κ)
π̂_BSPA = λ · SPA(test) + (1 − λ) · π_train
```

This reduces variance when the test set is small; as n grows, λ → 1 and the estimate converges to plain SPA. BSPA ranks **6th of 11 methods** in the APP benchmark (Friedman rank = 5.97, MSE = 0.01327), is **non-significantly different from Ens-PACC** (Nemenyi diff = 1.10 < CD = 2.76), and achieves **0.4 ms/sample inference** — 96× faster than Ens-PACC and 87× faster than KDEy-HD.

---

## Repository structure

```
├── quantifiers.py          # BSPA implementation (+ PACCFixed helper)
├── datasets.py             # Dataset loading utilities (QuaPy UCI collection)
├── benchmark.py            # Main APP benchmark (30 datasets × 11 methods)
├── methods.py              # Method definitions
├── analyse.py              # Result analysis and figure generation
├── plots.py                # Plotting helpers
├── ablation_kappa.py       # Ablation 1: BSPA sensitivity to κ
├── ablation_samplesize.py  # Ablation 2: sensitivity to test sample size
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation

```bash
git clone https://github.com/ceferra/bspa-emqspa-quantification.git
cd bspa-emqspa-quantification
pip install -r requirements.txt
```

Requires Python ≥ 3.9.

---

## Quick start

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from quantifiers import BSPAQuantifier

clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])

q = BSPAQuantifier(classifier=clf, val_split=0.4, kappa=5.0)
q.fit(X_train, y_train)
prevalence = q.predict(X_test)   # array([neg_prev, pos_prev])
```

---

## Reproducing the experiments

**Main benchmark** (30 datasets × 11 methods, APP protocol):
```bash
python benchmark.py
python analyse.py
```

**Ablation studies:**
```bash
python ablation_kappa.py        # κ sensitivity (saves ablation_kappa.pkl)
python ablation_samplesize.py   # sample size sensitivity (saves ablation_samplesize.pkl)
```

Results are saved as `.pkl` files and figures are written to `figures/`.

---

## Datasets

30 binary UCI datasets loaded via QuaPy's built-in collection (`fetch_UCIBinaryLabelledCollection`).  
Multi-class datasets are binarised by QuaPy (one-vs-rest by class index).  
Full description and experimental justification in §4 of the [paper](paper.pdf).

| # | Dataset ID | Full Name | N | Features | π(+) |
|---|-----------|-----------|--:|:--------:|-----:|
| 1 | balance.1 | Balance Scale | 625 | 4 | 0.46 |
| 2 | balance.3 | Balance Scale | 625 | 4 | 0.46 |
| 3 | breast-cancer | Breast Cancer Wisconsin | 683 | 9 | 0.65 |
| 4 | cmc.1 | Contraceptive Method Choice | 1473 | 9 | 0.43 |
| 5 | cmc.2 | Contraceptive Method Choice | 1473 | 9 | 0.23 |
| 6 | cmc.3 | Contraceptive Method Choice | 1473 | 9 | 0.35 |
| 7 | ctg.1 | Cardiotocography | 2126 | 21 | 0.78 |
| 8 | ctg.2 | Cardiotocography | 2126 | 21 | 0.14 |
| 9 | ctg.3 | Cardiotocography | 2126 | 21 | 0.08 |
| 10 | german | German Credit | 1000 | 24 | 0.70 |
| 11 | haberman | Haberman's Survival | 306 | 3 | 0.27 |
| 12 | ionosphere | Ionosphere | 351 | 34 | 0.36 |
| 13 | iris.1 | Iris | 150 | 4 | 0.33 |
| 14 | iris.2 | Iris | 150 | 4 | 0.33 |
| 15 | iris.3 | Iris | 150 | 4 | 0.33 |
| 16 | mammographic | Mammographic Mass | 830 | 5 | 0.49 |
| 17 | pageblocks.5 | Page Blocks | 5473 | 10 | 0.02 |
| 18 | semeion | Semeion Handwritten Digit | 1593 | 256 | 0.10 |
| 19 | sonar | Sonar | 208 | 60 | 0.47 |
| 20 | spambase | Spambase | 4601 | 57 | 0.39 |
| 21 | spectf | SPECTF Heart | 267 | 44 | 0.21 |
| 22 | tictactoe | Tic-Tac-Toe Endgame | 958 | 9 | 0.35 |
| 23 | transfusion | Blood Transfusion | 748 | 4 | 0.24 |
| 24 | wdbc | Wisconsin Diagnostic BC | 569 | 30 | 0.37 |
| 25 | wine.1 | Wine | 178 | 13 | 0.33 |
| 26 | wine.2 | Wine | 178 | 13 | 0.40 |
| 27 | wine.3 | Wine | 178 | 13 | 0.27 |
| 28 | wine-q-red | Wine Quality Red | 1599 | 11 | 0.54 |
| 29 | wine-q-white | Wine Quality White | 4898 | 11 | 0.67 |
| 30 | yeast | Yeast | 1484 | 8 | 0.29 |

**N** = total instances; **π(+)** = positive-class prevalence after binarisation.  
All datasets are publicly available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/).  
`balance.2` is excluded (not recognised by the current QuaPy version).

---

## Experimental protocol

Full details in §4 of the [paper](paper.pdf).

- **Splits**: 5 independent stratified 70/30 train/test splits per dataset
- **Classifier**: Logistic Regression with Standard Scaling (LR+SS)
- **Validation split**: 40% of training data held out for SPA calibration (`val_split=0.4`); all methods use `random_state=0` for reproducibility
- **Evaluation**: Artificial Prevalence Protocol (APP) — 21 prevalence levels × 10 repeats = 210 test samples per split, sample size n=100
- **Metric**: Mean Squared Error (MSE)
- **Statistical test**: Friedman + Nemenyi post-hoc (α=0.05, CD=2.76 for K=11, N=30)
- **Timing**: `time.perf_counter()`, single CPU core, mean over 30 datasets

---

## Main results

Full results, CD diagram, Pareto analysis, and ablation studies in §5–6 of the [paper](paper.pdf).

| Rank | Method | Avg Rank | Avg MSE | Fit (s) | Pred (ms/sample) |
|------|--------|:--------:|--------:|--------:|----------------:|
| 1 | KDEy-HD | 2.43 | 0.00987 | 0.316 | 33.6 |
| 2 | EMQ | 2.57 | 0.00966 | 0.005 | 0.6 |
| 3 | KDEy-ML | 3.17 | 0.01130 | 0.029 | 5.0 |
| 4 | Ens-PACC | 4.87 | 0.00989 | 0.073 | 37.2 |
| 5 | PACC | 5.87 | 0.01400 | 0.005 | 0.4 |
| **6** | **BSPA** ★ | **5.97** | **0.01327** | **0.006** | **0.4** |
| 7 | MS | 6.43 | 0.01277 | 0.015 | 0.2 |
| 8 | ACC | 7.73 | 0.02851 | 0.005 | 2.3 |
| 9 | CC | 8.63 | 0.06257 | 0.005 | 0.1 |
| 10 | HDy | 8.87 | 0.03932 | 0.006 | 8.9 |
| 11 | PCC | 9.47 | 0.04549 | 0.004 | 0.1 |

**BSPA is non-significantly different from Ens-PACC** (Nemenyi diff = 1.10 < CD = 2.76) while being 96× faster at inference.  
Despite ranking 6th, **BSPA achieves lower MSE than PACC** (0.01327 vs. 0.01400).

Friedman χ²(10) = 175.10, p = 2.44 × 10⁻³².

---

## License

MIT
