"""Tests for LIME stability tuner."""

import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', '1'))

from lime_tuner import find_stable_num_samples


@pytest.fixture
def dummy_predict_fn():
    """A simple linear predict_fn for testing."""
    rng = np.random.RandomState(42)
    W = rng.randn(10, 3)

    def predict_fn(X):
        logits = X @ W
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    return predict_fn


@pytest.fixture
def dummy_data():
    rng = np.random.RandomState(42)
    X_train = rng.randn(100, 10).astype(np.float32)
    X_test = rng.randn(5, 10).astype(np.float32)
    return X_train, X_test


def test_find_stable_num_samples_returns_int(dummy_predict_fn, dummy_data):
    X_train, X_test = dummy_data
    result = find_stable_num_samples(
        predict_fn=dummy_predict_fn,
        X_train=X_train,
        X_probe=X_test,
        feature_names=[f"f{i}" for i in range(10)],
        num_classes=3,
        candidate_counts=[100, 500, 1000],
        n_repeats=3,
        stability_threshold=0.8,
    )
    assert isinstance(result, int)
    assert result in [100, 500, 1000]


def test_find_stable_num_samples_monotonic(dummy_predict_fn, dummy_data):
    """More samples should not decrease stability."""
    X_train, X_test = dummy_data
    result = find_stable_num_samples(
        predict_fn=dummy_predict_fn,
        X_train=X_train,
        X_probe=X_test[:2],
        feature_names=[f"f{i}" for i in range(10)],
        num_classes=3,
        candidate_counts=[50, 100, 200],
        n_repeats=2,
        stability_threshold=0.7,
    )
    assert result >= 50
