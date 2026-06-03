"""
Method factory for the quantification benchmark.

Builds a fresh dict of {name: quantifier_instance} on each call.
All methods share a Scaled Logistic Regression base classifier —
the standard choice in recent quantification benchmarks (Moreo et al. 2021).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import quapy.method.aggregative as qpa

from quantifiers import BSPAQuantifier, EMQSPAQuantifier, PACCFixed

# --------------------------------------------------------------------------- #
# Optional dependencies — silently degrade if not available
# --------------------------------------------------------------------------- #
try:
    from quapy.method.aggregative import KDEyHD, KDEyML
    KDEY_OK = True
except ImportError:
    KDEY_OK = False

try:
    from quapy.method.meta import Ensemble
    ENSEMBLE_OK = True
except ImportError:
    ENSEMBLE_OK = False

# Tag sets used for colouring / labelling throughout the benchmark
OUR_METHODS   = {"BSPA", "EMQSPA"}
NEWQP_METHODS = {"KDEy-HD", "KDEy-ML", "Ens-PACC"}

# Shared hyper-parameter
VAL_SPLIT = 0.4   # fraction of training set held out for calibration


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def make_clf():
    """Return a fresh Scaled Logistic Regression pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0, random_state=0)),
    ])


def make_methods():
    """
    Build and return a fresh dict of {name: quantifier_instance}.

    Call once per train/test split so each quantifier starts unfitted.
    """
    m = {}

    # ── QuaPy built-in methods ──────────────────────────────────────────────
    m["CC"]   = qpa.CC(make_clf())
    m["PCC"]  = qpa.PCC(make_clf())
    m["ACC"]  = qpa.ACC(make_clf(),  val_split=VAL_SPLIT)
    m["PACC"] = PACCFixed(make_clf(), val_split=VAL_SPLIT)
    m["EMQ"]  = qpa.EMQ(make_clf())
    m["HDy"]  = qpa.HDy(make_clf(),  val_split=VAL_SPLIT)
    m["MS"]   = qpa.MS(make_clf(),   val_split=VAL_SPLIT)

    # KDEy (QuaPy >= 0.1.9)
    if KDEY_OK:
        m["KDEy-HD"] = KDEyHD(make_clf())
        m["KDEy-ML"] = KDEyML(make_clf())

    # Ensemble wrapping PACC
    if ENSEMBLE_OK:
        try:
            m["Ens-PACC"] = Ensemble(
                qpa.PACC(make_clf(), val_split=VAL_SPLIT),
                size=15,
                policy="ave",
            )
        except Exception as e:
            print(f"  Ensemble skipped: {e}")

    # ── Our custom proposals ────────────────────────────────────────────────
    m["BSPA"]   = BSPAQuantifier(make_clf(),   val_split=VAL_SPLIT, kappa=5.0)
    m["EMQSPA"] = EMQSPAQuantifier(make_clf(), val_split=VAL_SPLIT)

    return m


def method_names():
    """Return the list of method names produced by make_methods()."""
    return list(make_methods().keys())
