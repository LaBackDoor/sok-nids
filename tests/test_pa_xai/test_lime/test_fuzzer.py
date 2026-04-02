import numpy as np
import pytest


def _make_simple_schema():
    from pa_xai.core.schemas import DatasetSchema, HierarchicalConstraint

    return DatasetSchema(
        name="test",
        feature_names=["proto", "duration", "pkt_max", "pkt_mean", "pkt_min", "syn_flag"],
        protocol_feature="proto",
        non_negative_features=["duration", "pkt_max", "pkt_mean", "pkt_min", "syn_flag"],
        tcp_only_features=["syn_flag"],
        discrete_features=["syn_flag"],
        hierarchical_constraints=[
            HierarchicalConstraint("pkt_max", "pkt_mean", "pkt_min"),
        ],
        protocol_encoding="integer",
    )


def test_fuzzer_returns_correct_shape():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=500, sigma=0.1)
    assert neighborhood.shape == (500, 6)


def test_fuzzer_first_row_is_original():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=100, sigma=0.1)
    np.testing.assert_array_equal(neighborhood[0], x)


def test_fuzzer_enforces_non_negativity():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 1.0, 2.0, 1.0, 0.5, 1.0])
    neighborhood = fuzzer.generate(x, num_samples=5000, sigma=5.0)
    non_neg_cols = [1, 2, 3, 4, 5]
    assert np.all(neighborhood[:, non_neg_cols] >= 0)


def test_fuzzer_enforces_hierarchy():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=5000, sigma=2.0)
    assert np.all(neighborhood[:, 2] >= neighborhood[:, 3])
    assert np.all(neighborhood[:, 3] >= neighborhood[:, 4])


def test_fuzzer_zeros_tcp_features_for_udp():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([17.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)
    assert np.all(neighborhood[:, 5] == 0.0)


def test_fuzzer_preserves_tcp_features_for_tcp():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)
    assert not np.all(neighborhood[1:, 5] == 0.0)


def test_fuzzer_discrete_features_are_integers():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)
    assert np.all(neighborhood[:, 5] == np.round(neighborhood[:, 5]))


def test_fuzzer_protocol_column_fixed():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([17.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=5.0)
    assert np.all(neighborhood[:, 0] == 17.0)


def test_fuzzer_feature_wise_sigma():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    sigma = np.array([0.0, 10.0, 5.0, 3.0, 1.0, 0.5])
    neighborhood = fuzzer.generate(x, num_samples=5000, sigma=sigma)
    assert np.all(neighborhood[:, 0] == 6.0)
    assert np.std(neighborhood[1:, 1]) > 0


def test_fuzzer_passes_udp_icmp_labels():
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "udp_feat", "icmp_feat", "duration"],
        protocol_feature="proto",
        non_negative_features=["duration"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        udp_only_features=["udp_feat"],
        icmp_only_features=["icmp_feat"],
    )
    fuzzer = DomainConstraintFuzzer(schema)
    x_row = np.array([6.0, 5.0, 3.0, 100.0])  # TCP flow
    neighborhood = fuzzer.generate(x_row, num_samples=100, sigma=1.0)
    assert np.all(neighborhood[:, 1] == 0.0)  # udp_feat zeroed for TCP
    assert np.all(neighborhood[:, 2] == 0.0)  # icmp_feat zeroed for TCP


def test_fuzzer_conditioned_sampling_uses_training_distribution():
    """With X_train, discrete features are sampled per-protocol from training data."""
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()

    # TCP samples have syn_flag in {1, 2, 3}; UDP has syn_flag=0
    X_train = np.array([
        [6.0, 100.0, 50.0, 30.0, 10.0, 1.0],
        [6.0, 110.0, 60.0, 40.0, 20.0, 2.0],
        [6.0, 120.0, 70.0, 50.0, 30.0, 3.0],
        [17.0, 80.0, 40.0, 20.0, 5.0, 0.0],
        [17.0, 90.0, 45.0, 25.0, 8.0, 0.0],
    ], dtype=np.float32)

    fuzzer = DomainConstraintFuzzer(schema, X_train=X_train)
    x_tcp = np.array([6.0, 150.0, 80.0, 60.0, 40.0, 2.0], dtype=np.float32)
    neighborhood = fuzzer.generate(x_tcp, num_samples=1000, sigma=1.0)

    # Protocol is perturbed — some rows are TCP, some UDP
    tcp_rows = neighborhood[1:][neighborhood[1:, 0] == 6.0]
    udp_rows = neighborhood[1:][neighborhood[1:, 0] == 17.0]
    assert len(tcp_rows) > 0, "Should have some TCP rows"
    assert len(udp_rows) > 0, "Should have some UDP rows"

    # TCP rows: syn_flag sampled from {1, 2, 3}
    tcp_syn = np.unique(tcp_rows[:, 5])
    assert set(tcp_syn).issubset({1.0, 2.0, 3.0}), f"TCP syn_flag: {tcp_syn}"

    # UDP rows: syn_flag zeroed by enforcer (TCP-only feature)
    assert np.all(udp_rows[:, 5] == 0.0), "UDP syn_flag should be zeroed"


def test_fuzzer_perturbs_protocol_with_X_train():
    """With X_train, protocol is sampled from training distribution."""
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()

    X_train = np.array([
        [6.0, 100.0, 50.0, 30.0, 10.0, 1.0],
        [6.0, 110.0, 60.0, 40.0, 20.0, 2.0],
        [17.0, 80.0, 40.0, 20.0, 5.0, 0.0],
        [17.0, 90.0, 45.0, 25.0, 8.0, 0.0],
    ], dtype=np.float32)

    fuzzer = DomainConstraintFuzzer(schema, X_train=X_train)
    x = np.array([6.0, 150.0, 80.0, 60.0, 40.0, 2.0], dtype=np.float32)
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)

    # Protocol should have both TCP and UDP values (not all fixed)
    protos = np.unique(neighborhood[1:, 0])
    assert 6.0 in protos, "Should have TCP rows"
    assert 17.0 in protos, "Should have UDP rows"
    # No fractional protocol values
    assert set(protos).issubset({6.0, 17.0}), f"Invalid protocols: {protos}"


def test_fuzzer_without_X_train_holds_discrete_fixed():
    """Without X_train, discrete features stay at the input value."""
    from pa_xai.lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()

    fuzzer = DomainConstraintFuzzer(schema)  # no X_train
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0], dtype=np.float32)
    neighborhood = fuzzer.generate(x, num_samples=500, sigma=1.0)

    # syn_flag (discrete) should be held at 3.0 for all rows
    assert np.all(neighborhood[:, 5] == 3.0)
