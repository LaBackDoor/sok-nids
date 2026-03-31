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
