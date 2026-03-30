import numpy as np
import pytest


def _make_schema():
    from pa_xai.core.schemas import DatasetSchema
    return DatasetSchema(
        name="test",
        feature_names=["proto", "duration", "bytes"],
        protocol_feature="proto",
        non_negative_features=["duration", "bytes"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )


def _make_training_data():
    X_train = np.array([
        [6.0, 100.0, 500.0],
        [6.0, 110.0, 510.0],
        [17.0, 50.0, 200.0],
        [6.0, 999.0, 9999.0],
        [17.0, 800.0, 8000.0],
        [6.0, 200.0, 600.0],
    ])
    y_train = np.array([0, 0, 0, 1, 1, 0])
    return X_train, y_train


def test_baseline_matches_protocol():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    x_tcp_attack = np.array([6.0, 999.0, 9999.0])
    baseline = get_protocol_valid_baseline(x_tcp_attack, X_train, y_train, schema)
    assert baseline[0] == 6.0


def test_baseline_is_benign():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    x_attack = np.array([6.0, 999.0, 9999.0])
    baseline = get_protocol_valid_baseline(x_attack, X_train, y_train, schema)
    matches = np.where(np.all(X_train == baseline, axis=1))[0]
    assert len(matches) > 0
    assert y_train[matches[0]] == 0


def test_baseline_is_nearest_neighbor():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    x_attack = np.array([6.0, 115.0, 515.0])
    baseline = get_protocol_valid_baseline(x_attack, X_train, y_train, schema)
    assert baseline[1] == 110.0
    assert baseline[2] == 510.0


def test_baseline_udp_never_returns_tcp():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    x_udp_attack = np.array([17.0, 800.0, 8000.0])
    baseline = get_protocol_valid_baseline(x_udp_attack, X_train, y_train, schema)
    assert baseline[0] == 17.0


def test_baseline_median_k_strategy():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    x_attack = np.array([6.0, 150.0, 550.0])
    baseline = get_protocol_valid_baseline(
        x_attack, X_train, y_train, schema,
        top_k=3, strategy="median_k",
    )
    assert baseline[0] == 6.0
    assert baseline[1] == 110.0


def test_baseline_raises_when_no_matching_benign():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    x_icmp = np.array([1.0, 50.0, 100.0])
    with pytest.raises(ValueError, match="No benign samples"):
        get_protocol_valid_baseline(x_icmp, X_train, y_train, schema)


def test_baseline_custom_benign_label():
    from pa_xai.core.baseline import get_protocol_valid_baseline
    schema = _make_schema()
    X_train, y_train = _make_training_data()
    y_flipped = 1 - y_train
    x_attack = np.array([6.0, 999.0, 9999.0])
    baseline = get_protocol_valid_baseline(
        x_attack, X_train, y_flipped, schema, benign_label=0,
    )
    assert baseline[1] == 999.0
