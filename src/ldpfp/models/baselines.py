"""Phase 6c: Gradient-boosting and tree-based baseline classifiers."""
from __future__ import annotations

import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def build_baselines(n_jobs: int = -1) -> dict:
    """
    Construct the non-deep baseline models used for comparison
    with the hierarchical neural classifier.

    Each boosting model is wrapped in MultiOutputClassifier so that
    an independent binary classifier is learned for every GO term.
    """

    return {
        "xgboost": MultiOutputClassifier(
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.1,
                tree_method="hist",
                n_jobs=n_jobs,
                eval_metric="logloss",
            ),
            n_jobs=n_jobs,
        ),

        "lightgbm": MultiOutputClassifier(
            LGBMClassifier(
                n_estimators=300,
                num_leaves=31,
                learning_rate=0.1,
                n_jobs=n_jobs,
                verbose=-1,
            ),
            n_jobs=n_jobs,
        ),

        "catboost": MultiOutputClassifier(
            CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.1,
                verbose=False,
                allow_writing_files=False,
            ),
            n_jobs=n_jobs,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=n_jobs,
            random_state=42,
        ),
    }


def fit_and_predict(
    model,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Fit a multi-label baseline and return positive-class
    probabilities for every GO term.
    """

    model.fit(X_train, Y_train)

    if not hasattr(model, "predict_proba"):
        return model.predict(X_test).astype(np.float32)

    proba = model.predict_proba(X_test)

    # MultiOutputClassifier returns one probability matrix
    # per GO label.
    if isinstance(proba, list):
        outputs = []

        for p in proba:
            # Normally binary classification gives (n_samples, 2).
            if p.shape[1] == 2:
                outputs.append(p[:, 1])

            # A label may contain only one class in a small/training
            # subset. Handle it safely rather than crashing.
            elif p.shape[1] == 1:
                classes = getattr(model, "classes_", None)

                # For our eventual real pipeline these degenerate
                # labels should normally be removed before fitting.
                outputs.append(np.zeros(p.shape[0], dtype=np.float32))

            else:
                raise ValueError(
                    f"Unexpected predict_proba shape: {p.shape}"
                )

        return np.stack(outputs, axis=1)

    return np.asarray(proba)