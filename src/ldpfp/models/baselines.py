"""Phase 7B: Tree-based baseline classifiers.

Each GO term is trained independently so that:
- single-class labels can be handled explicitly,
- training can be checkpointed/resumed,
- prediction matrices retain the full GO vocabulary.
"""
from __future__ import annotations

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


BASELINE_NAMES = (
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
)


def build_baseline(
    name: str,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """Construct one binary baseline classifier."""

    name = name.lower().strip()

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            n_jobs=n_jobs,
            eval_metric="logloss",
            random_state=random_state,
        )

    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.1,
            n_jobs=n_jobs,
            verbose=-1,
            random_state=random_state,
        )

    if name == "catboost":
        return CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            thread_count=n_jobs,
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown baseline {name!r}. "
        f"Expected one of {BASELINE_NAMES}."
    )