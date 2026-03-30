import numpy as np
import pytest


def _make_simple_schema():
    from pa_lime.schemas import DatasetSchema, HierarchicalConstraint

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
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=500, sigma=0.1)
    assert neighborhood.shape == (500, 6)


def test_fuzzer_first_row_is_original():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=100, sigma=0.1)
    np.testing.assert_array_equal(neighborhood[0], x)


def test_fuzzer_enforces_non_negativity():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 1.0, 2.0, 1.0, 0.5, 1.0])
    neighborhood = fuzzer.generate(x, num_samples=5000, sigma=5.0)
    non_neg_cols = [1, 2, 3, 4, 5]
    assert np.all(neighborhood[:, non_neg_cols] >= 0)


def test_fuzzer_enforces_hierarchy():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=5000, sigma=2.0)
    assert np.all(neighborhood[:, 2] >= neighborhood[:, 3])
    assert np.all(neighborhood[:, 3] >= neighborhood[:, 4])


def test_fuzzer_zeros_tcp_features_for_udp():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([17.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)
    assert np.all(neighborhood[:, 5] == 0.0)


def test_fuzzer_preserves_tcp_features_for_tcp():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)
    assert not np.all(neighborhood[1:, 5] == 0.0)


def test_fuzzer_discrete_features_are_integers():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=1.0)
    assert np.all(neighborhood[:, 5] == np.round(neighborhood[:, 5]))


def test_fuzzer_protocol_column_fixed():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([17.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    neighborhood = fuzzer.generate(x, num_samples=1000, sigma=5.0)
    assert np.all(neighborhood[:, 0] == 17.0)


def test_fuzzer_feature_wise_sigma():
    from pa_lime.fuzzer import DomainConstraintFuzzer
    schema = _make_simple_schema()
    fuzzer = DomainConstraintFuzzer(schema)
    x = np.array([6.0, 100.0, 50.0, 30.0, 10.0, 3.0])
    sigma = np.array([0.0, 10.0, 5.0, 3.0, 1.0, 0.5])
    neighborhood = fuzzer.generate(x, num_samples=5000, sigma=sigma)
    assert np.all(neighborhood[:, 0] == 6.0)
    assert np.std(neighborhood[1:, 1]) > 0
