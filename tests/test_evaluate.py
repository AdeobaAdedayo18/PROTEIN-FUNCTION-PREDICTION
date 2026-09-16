import numpy as np
from ldpfp.evaluate import fmax_score, hierarchy_violation_rate

def test_fmax_perfect_prediction():
    y_true = np.array([[1, 0], [0, 1]])
    y_prob = np.array([[0.9, 0.1], [0.1, 0.9]])
    f, t = fmax_score(y_true, y_prob)
    assert f == 1.0

def test_hierarchy_violation_rate_detects_violation():
    y_prob = np.array([[0.9, 0.1]])  # child=col0 prob 0.9 > parent=col1 prob 0.1 -> violation
    rate = hierarchy_violation_rate(y_prob, [(0, 1)])
    assert rate == 1.0
def test_macro_aupr_ignores_labels_without_positives():
    from ldpfp.evaluate import aupr_scores

    # Third GO term has no positive examples.
    y_true = np.array([
        [1, 0, 0],
        [0, 1, 0],
    ])

    y_prob = np.array([
        [0.9, 0.1, 0.2],
        [0.1, 0.9, 0.3],
    ])

    scores = aupr_scores(
        y_true,
        y_prob,
    )

    assert scores["aupr_micro"] == 1.0
    assert scores["aupr_macro"] == 1.0
    assert scores["macro_evaluable_terms"] == 2
    assert scores["macro_total_terms"] == 3