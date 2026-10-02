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

ACCELERATORS = ("cpu", "cuda")


def build_baseline(
    name: str,
    n_jobs: int = -1,
    random_state: int = 42,
    accelerator: str = "cpu",
):
    """Construct one binary baseline classifier.

    accelerator="cuda" enables GPU training for XGBoost and CatBoost.
    LightGBM and Random Forest remain on CPU in this project pipeline.
    """
    name = name.lower().strip()
    accelerator = accelerator.lower().strip()

    if accelerator not in ACCELERATORS:
        raise ValueError(
            f"Unknown accelerator {accelerator!r}. Expected one of {ACCELERATORS}."
        )

    if name == "xgboost":
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            tree_method="hist",
            device="cuda" if accelerator == "cuda" else "cpu",
            n_jobs=n_jobs,
            eval_metric="logloss",
            random_state=random_state,
        )

    if name == "lightgbm":
        if accelerator == "cuda":
            raise ValueError(
                "CUDA mode is not enabled for the LightGBM baseline in this pipeline. "
                "Use --accelerator cpu."
            )
        return LGBMClassifier(
            n_estimators=300,
            num_leaves=31,
            learning_rate=0.1,
            n_jobs=n_jobs,
            verbose=-1,
            random_state=random_state,
        )

    if name == "catboost":
        kwargs = dict(
            iterations=300,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            allow_writing_files=False,
            random_seed=random_state,
            thread_count=n_jobs,
            loss_function="Logloss",
        )
        if accelerator == "cuda":
            kwargs.update(task_type="GPU", devices="0")
        else:
            kwargs.update(task_type="CPU")
        return CatBoostClassifier(**kwargs)

    if name == "random_forest":
        if accelerator == "cuda":
            raise ValueError(
                "scikit-learn RandomForestClassifier is CPU-only in this pipeline. "
                "Use --accelerator cpu."
            )
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown baseline {name!r}. Expected one of {BASELINE_NAMES}."
    )
