import numpy as np

from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

from ldpfp.models.baselines import (
    build_baselines,
    fit_and_predict,
)


def test_build_baselines():
    models = build_baselines(n_jobs=1)

    assert "xgboost" in models
    assert "lightgbm" in models
    assert "catboost" in models
    assert "random_forest" in models


def test_fit_and_predict_shape():
    rng = np.random.default_rng(42)

    X_train = rng.normal(size=(30, 8)).astype(np.float32)
    X_test = rng.normal(size=(5, 8)).astype(np.float32)

    # Three synthetic GO labels, each containing both classes.
    Y_train = np.zeros((30, 3), dtype=np.int64)

    Y_train[:15, 0] = 1
    Y_train[::2, 1] = 1
    Y_train[::3, 2] = 1

    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=2,
            max_depth=2,
            tree_method="hist",
            n_jobs=1,
            eval_metric="logloss",
        ),
        n_jobs=1,
    )

    probs = fit_and_predict(
        model,
        X_train,
        Y_train,
        X_test,
    )

    assert probs.shape == (5, 3)
    assert np.all(probs >= 0)
    assert np.all(probs <= 1)