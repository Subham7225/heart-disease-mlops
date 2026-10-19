import numpy as np

from src.evaluate import confidence_from_proba, evaluate_binary_classifier


def test_evaluate_binary_classifier_perfect_scores():
    y = [0, 1, 0, 1]
    metrics = evaluate_binary_classifier(y, y, [0.1, 0.9, 0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_confidence_is_max_probability():
    assert confidence_from_proba(np.array([0.3, 0.7])) == 0.7
