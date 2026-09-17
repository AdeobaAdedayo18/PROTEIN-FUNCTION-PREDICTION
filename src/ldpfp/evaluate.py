"""Phase 7: CAFA-style evaluation metrics + hierarchy-violation rate."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score

def precision_recall_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, t: float):
    y_pred = (y_prob >= t).astype(int)
    tp = (y_pred * y_true).sum(axis=1)
    pred_pos = y_pred.sum(axis=1)
    true_pos = y_true.sum(axis=1)
    covered = pred_pos > 0
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp, dtype=float), where=true_pos > 0)
    if covered.sum() == 0:
        return 0.0, 0.0
    return precision[covered].mean(), recall[true_pos > 0].mean()

def fmax_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds=None,
) -> tuple[float, float]:
    """CAFA-style protein-centric Fmax."""

    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    best_f, best_t = 0.0, 0.0

    for t in thresholds:
        p, r = precision_recall_at_threshold(
            y_true,
            y_prob,
            t,
        )

        f = (
            0.0
            if (p + r) == 0
            else 2 * p * r / (p + r)
        )

        if f > best_f:
            best_f = f
            best_t = float(t)

    return float(best_f), float(best_t)


def aupr_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict:
    """
    Compute micro- and macro-averaged AUPR.

    Micro-AUPR pools all protein-label prediction pairs.

    Macro-AUPR is calculated only over GO terms that contain at least
    one positive example in the evaluated split. Terms with zero
    positives cannot have a meaningful per-label precision-recall
    curve and are therefore excluded from the macro average.
    """

    # ---------------------------------------------------------
    # Micro-AUPR
    # ---------------------------------------------------------
    micro = average_precision_score(
        y_true,
        y_prob,
        average="micro",
    )

    # ---------------------------------------------------------
    # Macro-AUPR
    # ---------------------------------------------------------
    # A GO term is evaluable only if it has at least one
    # positive example in this evaluation split.
    positive_mask = y_true.sum(axis=0) > 0

    n_evaluable_terms = int(positive_mask.sum())
    n_total_terms = int(y_true.shape[1])

    if n_evaluable_terms == 0:
        macro = 0.0

    else:
        per_label_ap = average_precision_score(
            y_true[:, positive_mask],
            y_prob[:, positive_mask],
            average=None,
        )

        macro = float(
            np.mean(per_label_ap)
        )

    return {
        "aupr_micro": float(micro),
        "aupr_macro": macro,
        "macro_evaluable_terms": n_evaluable_terms,
        "macro_total_terms": n_total_terms,
    }
def hierarchy_violation_rate(y_prob: np.ndarray, parent_child_pairs: list[tuple[int, int]]) -> float:
    """Fraction of (protein, child-parent pair) instances where child_prob > parent_prob.
    Reported before AND after the post-hoc consistency pass (Section 3.6) to quantify
    the concrete benefit of hierarchical GO-awareness — this is your headline novelty metric."""
    if not parent_child_pairs:
        return 0.0
    child_idx = [p[0] for p in parent_child_pairs]
    parent_idx = [p[1] for p in parent_child_pairs]
    violations = (y_prob[:, child_idx] > y_prob[:, parent_idx]).mean()
    return float(violations)

def full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    parent_child_pairs: list[tuple[int, int]],
) -> dict:

    fmax, t_star = fmax_score(
        y_true,
        y_prob,
    )

    aupr = aupr_scores(
        y_true,
        y_prob,
    )

    hvr = hierarchy_violation_rate(
        y_prob,
        parent_child_pairs,
    )

    return {
        "fmax": fmax,
        "best_threshold": t_star,
        **aupr,
        "hierarchy_violation_rate": hvr,
    }
