# BSPA and EMQSPA: Competitive Quantification at Low Computational Cost

Code and experiments for the paper:

> **BSPA and EMQSPA: Competitive Quantification at Low Computational Cost**  
> Cèsar Ferri et al. *Workshop on Learning to Quantify (LQ @ ECML-PKDD 2025)* — **under review**.  
> 📄 [paper.pdf](paper.pdf)

---

## Overview

This repository contains two lightweight quantifiers for binary prevalence estimation under label shift, built on top of [QuaPy](https://github.com/HLT-ISTI/QuaPy), and all the code needed to reproduce the experiments reported in the paper.

- **BSPA** (*Beta-Smoothed Scaled Probability Average*, §3.1 of the paper): shrinks the SPA estimate toward the training prior with a single hyperparameter κ, providing explicit variance control at small test-set sizes.
- **EMQSPA** (*EMQ warm-started from SPA*, §3.2 of the paper): initialises the Saerens et al. (2002) EM quantifier from the SPA estimate instead of the plain probability average, reducing EM iteration count while converging to the same fixed point.

Both methods achieve **statistically indistinguishable accuracy from the best-ranked KDEy-HD** (Nemenyi CD = 3.04; rank gap = 2.53) while being **59× faster to train** and **50× faster at inference** (see §5 of the paper).

---

## Repository structure

```
├── quantifiers.py          # BSPA and EMQSPA implementations
├── datasets.py             # Dataset loading utilities (QuaPy UCI collection)
├── benchmark.py            # Main APP benchmark (30 datasets × 12 methods)
├── methods.py              # Baseline method definitions
├── analyse.py              # Result analysis and figure generation
├── plots.py                # Plotting helpers
├── ablation_kappa.py       # Ablation 1: BSPA sensitivity to κ
├── ablation_samplesize.py  # Ablation 2: sensitivity to test sample size
├── ablation_convergence.py # Ablation 3: EMQ vs EMQSPA iteration count
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
from quantifiers import BSPAQuantifier, EMQSPAQuantifier

clf = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])

# BSPA
q = BSPAQuantifier(classifier=clf, val_split=0.4, kappa=5.0)
q.fit(X_train, y_train)
prevalence = q.predict(X_test)   # array([neg_prev, pos_prev])

# EMQSPA
q = EMQSPAQuantifier(classifier=clf, val_split=0.4)
q.fit(X_train, y_train)
prevalence = q.predict(X_test)
```

---

## Reproducing the experiments

**Main benchmark** (30 datasets × 12 methods, APP protocol):
```bash
python benchmark.py
python analyse.py
```

**Ablation studies:**
```bash
python ablation_kappa.py        # κ sensitivity (saves ablation_kappa.pkl)
python ablation_samplesize.py   # sample size sensitivity (saves ablation_samplesize.pkl)
python ablation_convergence.py  # EM convergence (saves ablation_convergence.pkl)
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
- **Validation split**: 40% of training data held out for SPA calibration (val_split=0.4)
- **Evaluation**: Artificial Prevalence Protocol (APP) — 21 prevalence levels × 10 repeats = 210 test samples per split, sample size n=100
- **Metric**: Mean Squared Error (MSE)
- **Statistical test**: Friedman + Nemenyi post-hoc (α=0.05, CD=3.04 for K=12, N=30)
- **Timing**: `time.perf_counter()`, single CPU core, mean over 30 datasets

---

## Main results

Full results, CD diagram, Pareto analysis, and ablation studies in §5–6 of the [paper](paper.pdf).

| Rank | Method | Avg MSE | Fit time (s) | Predict time (ms/sample) |
|------|--------|--------:|-------------:|------------------------:|
| 1 | KDEy-HD | 0.00987 | 0.315 | 33.2 |
| 2 | EMQ | 0.00966 | 0.005 | 0.62 |
| 3 | Ens-PACC | 0.00990 | 0.074 | 37.4 |
| 4 | KDEy-ML | 0.01130 | 0.030 | 5.04 |
| **5** | **EMQSPA** | **0.01222** | **0.005** | **0.67** |
| **6** | **BSPA** | **0.01327** | **0.006** | **0.39** |
| 7 | PACC | 0.01344 | 0.005 | 2.27 |
| ... | ... | ... | ... | ... |

BSPA and EMQSPA are **not significantly different from KDEy-HD** (Nemenyi test, rank gap = 2.53 < CD = 3.04) while being 59× faster to train.

---

## License

MIT
