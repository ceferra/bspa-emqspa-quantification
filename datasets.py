"""
Dataset utilities for the quantification benchmark.

Loads 31 binary UCI datasets via QuaPy's built-in collection.
Excludes acute.a and acute.b (120 instances each, documented as
problematic in the QuaPy manual).
"""

import quapy as qp
from tqdm import tqdm

# Full list of 31 binary UCI datasets available in QuaPy
UCI_NAMES = [
    "balance.1", "balance.2", "balance.3",
    "breast-cancer",
    "cmc.1", "cmc.2", "cmc.3",
    "ctg.1", "ctg.2", "ctg.3",
    "german", "haberman", "ionosphere",
    "iris.1", "iris.2", "iris.3",
    "mammographic", "pageblocks.5", "semeion",
    "sonar", "spambase", "spectf", "tictactoe",
    "transfusion", "wdbc",
    "wine.1", "wine.2", "wine.3",
    "wine-q-red", "wine-q-white", "yeast",
]


def load_datasets(names=None, verbose=True):
    """
    Load UCI binary datasets from QuaPy's built-in collection.

    Parameters
    ----------
    names : list of str, optional
        Subset of dataset names to load. Defaults to all 31.
    verbose : bool
        Show tqdm progress bar.

    Returns
    -------
    datasets : dict
        {name: LabelledCollection}
    skipped : list of str
        Names of datasets that failed to load.
    """
    if names is None:
        names = UCI_NAMES

    datasets = {}
    skipped = []

    iterator = tqdm(names, desc="Loading datasets") if verbose else names
    for name in iterator:
        try:
            lc = qp.datasets.fetch_UCIBinaryLabelledCollection(name, verbose=False)
            datasets[name] = lc
        except Exception as e:
            skipped.append(name)
            if verbose:
                tqdm.write(f"  Skipped {name}: {e}")

    if verbose:
        print(f"Loaded {len(datasets)} datasets  ({len(skipped)} skipped: {skipped})")

    return datasets, skipped


def dataset_summary(datasets):
    """Return a pandas DataFrame summarising dataset statistics."""
    import pandas as pd

    rows = []
    for name, lc in datasets.items():
        prev = lc.prevalence()
        rows.append(
            {
                "Dataset": name,
                "N": len(lc),
                "features": lc.X.shape[1],
                "pi(-)": f"{prev[0]:.2f}",
                "pi(+)": f"{prev[1]:.2f}",
            }
        )
    return pd.DataFrame(rows).set_index("Dataset")
