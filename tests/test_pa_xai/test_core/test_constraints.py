import numpy as np
import pytest


def test_non_negativity_clamps_negative_values():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["duration", "bytes", "flag"],
        protocol_feature=None,
        non_negative_features=["duration", "bytes"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [-5.0, 10.0, -1.0],
        [3.0, -2.0, 0.5],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 0] == 0.0
    assert result[0, 1] == 10.0
    assert result[0, 2] == -1.0  # NOT in non_negative list
    assert result[1, 0] == 3.0
    assert result[1, 1] == 0.0
    assert result[1, 2] == 0.5


def test_hierarchical_constraint_enforces_max_ge_mean_ge_min():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, HierarchicalConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["pkt_max", "pkt_mean", "pkt_min"],
        protocol_feature=None,
        non_negative_features=["pkt_max", "pkt_mean", "pkt_min"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[
            HierarchicalConstraint("pkt_max", "pkt_mean", "pkt_min"),
        ],
        protocol_encoding="integer",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [2.0, 1.0, 5.0],
        [10.0, 3.0, 7.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 1] >= result[0, 2]
    assert result[0, 0] >= result[0, 1]
    assert result[1, 1] >= result[1, 2]
    assert result[1, 0] >= result[1, 1]


def test_discrete_features_are_rounded():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["flag_count", "continuous"],
        protocol_feature=None,
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=["flag_count"],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [2.7, 3.14],
        [0.3, 1.99],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 0] == 3.0
    assert result[0, 1] == 3.14
    assert result[1, 0] == 0.0
    assert result[1, 1] == 1.99


def test_tcp_only_zeroed_for_udp_integer_encoding():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "syn_flag", "window_bytes", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=["syn_flag", "window_bytes"],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [17.0, 5.0, 1024.0, 100.0],
        [17.0, 3.0, 512.0, 200.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=17.0, protocol_encoding="integer")
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0
    assert result[0, 3] == 100.0
    assert result[1, 1] == 0.0
    assert result[1, 2] == 0.0


def test_tcp_only_preserved_for_tcp_integer_encoding():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "syn_flag", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=["syn_flag"],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [6.0, 5.0, 100.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=6.0, protocol_encoding="integer")
    assert result[0, 1] == 5.0


def test_tcp_only_zeroed_for_non_tcp_string_encoding():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "urgent", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=["urgent"],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="string",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [0.0, 5.0, 100.0],
    ])
    result = enforcer.enforce(
        neighborhood, protocol_value=0.0, protocol_encoding="string", tcp_label_value=2
    )
    assert result[0, 1] == 0.0


def test_protocol_column_not_perturbed():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [17.5, 100.0],
        [6.3, 200.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=17.0, protocol_encoding="integer")
    assert result[0, 0] == 17.0
    assert result[1, 0] == 17.0


def test_all_constraints_applied_together():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, HierarchicalConstraint

    schema = DatasetSchema(
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
    enforcer = ConstraintEnforcer(schema)

    neighborhood = np.array([
        [17.0, -3.0, 2.0, 5.0, 1.0, 2.7],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=17.0, protocol_encoding="integer")
    assert result[0, 0] == 17.0
    assert result[0, 1] == 0.0
    assert result[0, 4] <= result[0, 3] <= result[0, 2]
    assert result[0, 5] == 0.0


def test_bounded_range_clamps_to_bounds():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, BoundedRangeConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["port", "rate", "other"],
        protocol_feature=None,
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        bounded_range_constraints=[
            BoundedRangeConstraint("port", 0.0, 65535.0),
            BoundedRangeConstraint("rate", 0.0, 1.0),
        ],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([
        [70000.0, 1.5, 42.0],
        [-100.0, -0.3, 99.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 0] == 65535.0
    assert result[0, 1] == 1.0
    assert result[0, 2] == 42.0
    assert result[1, 0] == 0.0
    assert result[1, 1] == 0.0


def test_std_range_clamped_to_max_minus_min():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, StdRangeConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["pkt_max", "pkt_std", "pkt_min"],
        protocol_feature=None,
        non_negative_features=["pkt_max", "pkt_std", "pkt_min"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        std_range_constraints=[
            StdRangeConstraint("pkt_std", "pkt_max", "pkt_min"),
        ],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([
        [10.0, 20.0, 5.0],
        [100.0, 3.0, 90.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 1] == 5.0
    assert result[1, 1] == 3.0


def test_cross_feature_ratio_recomputed():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, CrossFeatureConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["pkts", "duration", "pkts_per_s"],
        protocol_feature=None,
        non_negative_features=["pkts", "duration", "pkts_per_s"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        cross_feature_constraints=[
            CrossFeatureConstraint("pkts_per_s", "ratio", ["pkts", "duration"]),
        ],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([
        [100.0, 10.0, 999.0],
        [50.0, 0.0, 999.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 2] == pytest.approx(10.0)
    assert result[1, 2] == 0.0


def test_cross_feature_sum_ratio_recomputed():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, CrossFeatureConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["fwd_bytes", "bwd_bytes", "duration", "bytes_per_s"],
        protocol_feature=None,
        non_negative_features=["fwd_bytes", "bwd_bytes", "duration", "bytes_per_s"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        cross_feature_constraints=[
            CrossFeatureConstraint("bytes_per_s", "sum_ratio", ["fwd_bytes", "bwd_bytes", "duration"]),
        ],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([
        [300.0, 200.0, 5.0, 999.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 3] == pytest.approx(100.0)


def test_cross_feature_square_recomputed():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, CrossFeatureConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["pkt_std", "pkt_var"],
        protocol_feature=None,
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        cross_feature_constraints=[
            CrossFeatureConstraint("pkt_var", "square", ["pkt_std"]),
        ],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[4.0, 999.0]])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 1] == pytest.approx(16.0)


def test_cross_feature_equal_recomputed():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema, CrossFeatureConstraint

    schema = DatasetSchema(
        name="test",
        feature_names=["total_fwd", "subflow_fwd"],
        protocol_feature=None,
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        cross_feature_constraints=[
            CrossFeatureConstraint("subflow_fwd", "equal", ["total_fwd"]),
        ],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[42.0, 999.0]])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 1] == 42.0


def test_udp_only_zeroed_for_tcp():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "udp_feat", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        udp_only_features=["udp_feat"],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[6.0, 5.0, 100.0]])
    result = enforcer.enforce(neighborhood, protocol_value=6.0, protocol_encoding="integer")
    assert result[0, 1] == 0.0


def test_udp_only_preserved_for_udp():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "udp_feat", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        udp_only_features=["udp_feat"],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[17.0, 5.0, 100.0]])
    result = enforcer.enforce(neighborhood, protocol_value=17.0, protocol_encoding="integer")
    assert result[0, 1] == 5.0


def test_icmp_only_zeroed_for_tcp():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "icmp_feat", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        icmp_only_features=["icmp_feat"],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[6.0, 5.0, 100.0]])
    result = enforcer.enforce(neighborhood, protocol_value=6.0, protocol_encoding="integer")
    assert result[0, 1] == 0.0


def test_connection_only_zeroed_for_icmp():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "num_failed_logins", "logged_in", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        connection_only_features=["num_failed_logins", "logged_in"],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[1.0, 5.0, 1.0, 100.0]])
    result = enforcer.enforce(neighborhood, protocol_value=1.0, protocol_encoding="integer")
    assert result[0, 1] == 0.0
    assert result[0, 2] == 0.0
    assert result[0, 3] == 100.0


def test_connection_only_preserved_for_tcp():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["proto", "num_failed_logins", "duration"],
        protocol_feature="proto",
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        connection_only_features=["num_failed_logins"],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([[6.0, 5.0, 100.0]])
    result = enforcer.enforce(neighborhood, protocol_value=6.0, protocol_encoding="integer")
    assert result[0, 1] == 5.0


def test_duplicate_features_enforced_equal():
    from pa_xai.core.constraints import ConstraintEnforcer
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["hdr_len", "hdr_len_dup", "other"],
        protocol_feature=None,
        non_negative_features=[],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        duplicate_features=[("hdr_len", "hdr_len_dup")],
    )
    enforcer = ConstraintEnforcer(schema)
    neighborhood = np.array([
        [100.0, 999.0, 42.0],
        [200.0, 50.0, 10.0],
    ])
    result = enforcer.enforce(neighborhood, protocol_value=None, protocol_encoding="integer")
    assert result[0, 1] == 100.0
    assert result[1, 1] == 200.0
