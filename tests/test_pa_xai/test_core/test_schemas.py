# tests/test_pa_xai/test_core/test_schemas.py
import pytest
import numpy as np


def test_dataset_schema_stores_feature_metadata():
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["duration", "protocol", "src_bytes"],
        protocol_feature="protocol",
        non_negative_features=["duration", "src_bytes"],
        tcp_only_features=[],
        discrete_features=["protocol"],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    assert schema.name == "test"
    assert schema.feature_names == ["duration", "protocol", "src_bytes"]
    assert schema.protocol_feature == "protocol"
    assert schema.protocol_encoding == "integer"


def test_dataset_schema_resolves_feature_indices():
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["duration", "protocol", "src_bytes"],
        protocol_feature="protocol",
        non_negative_features=["duration", "src_bytes"],
        tcp_only_features=[],
        discrete_features=["protocol"],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    assert schema.protocol_index == 1
    assert schema.non_negative_indices == [0, 2]
    assert schema.discrete_indices == [1]


def test_dataset_schema_missing_protocol_feature_returns_none():
    from pa_xai.core.schemas import DatasetSchema

    schema = DatasetSchema(
        name="test",
        feature_names=["duration", "src_bytes"],
        protocol_feature=None,
        non_negative_features=["duration", "src_bytes"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    assert schema.protocol_index is None


def test_hierarchical_constraint_stores_max_mean_min():
    from pa_xai.core.schemas import HierarchicalConstraint

    hc = HierarchicalConstraint(
        max_feature="Fwd Packet Length Max",
        mean_feature="Fwd Packet Length Mean",
        min_feature="Fwd Packet Length Min",
    )
    assert hc.max_feature == "Fwd Packet Length Max"
    assert hc.mean_feature == "Fwd Packet Length Mean"
    assert hc.min_feature == "Fwd Packet Length Min"


def test_cic_ids_2017_schema_exists():
    from pa_xai.core.schemas import CIC_IDS_2017

    assert CIC_IDS_2017.name == "CIC-IDS-2017"
    assert "Flow Duration" in CIC_IDS_2017.feature_names
    assert CIC_IDS_2017.protocol_feature == "Protocol"
    assert CIC_IDS_2017.protocol_encoding == "integer"
    assert "FIN Flag Count" in CIC_IDS_2017.tcp_only_features
    assert "Init_Win_bytes_forward" in CIC_IDS_2017.tcp_only_features


def test_cse_cic_ids2018_schema_exists():
    from pa_xai.core.schemas import CSE_CIC_IDS2018

    assert CSE_CIC_IDS2018.name == "CSE-CIC-IDS2018"
    assert "Flow Duration" in CSE_CIC_IDS2018.feature_names
    assert CSE_CIC_IDS2018.protocol_feature == "Protocol"
    assert CSE_CIC_IDS2018.protocol_encoding == "integer"
    assert "FIN Flag Cnt" in CSE_CIC_IDS2018.tcp_only_features
    assert "Init Fwd Win Byts" in CSE_CIC_IDS2018.tcp_only_features


def test_nsl_kdd_schema_exists():
    from pa_xai.core.schemas import NSL_KDD

    assert NSL_KDD.name == "NSL-KDD"
    assert "duration" in NSL_KDD.feature_names
    assert NSL_KDD.protocol_feature == "protocol_type"
    assert NSL_KDD.protocol_encoding == "string"


def test_unsw_nb15_native_schema_exists():
    from pa_xai.core.schemas import UNSW_NB15_NATIVE

    assert UNSW_NB15_NATIVE.name == "UNSW-NB15-Native"
    assert "dur" in UNSW_NB15_NATIVE.feature_names
    assert UNSW_NB15_NATIVE.protocol_feature == "proto"
    assert "tcprtt" in UNSW_NB15_NATIVE.tcp_only_features
    assert "synack" in UNSW_NB15_NATIVE.tcp_only_features
    assert "ackdat" in UNSW_NB15_NATIVE.tcp_only_features


def test_unsw_nb15_cicflowmeter_schema_exists():
    from pa_xai.core.schemas import UNSW_NB15_CIC

    assert UNSW_NB15_CIC.name == "UNSW-NB15-CICFlowMeter"
    assert "Flow Duration" in UNSW_NB15_CIC.feature_names
    assert UNSW_NB15_CIC.protocol_feature == "Protocol"
    assert UNSW_NB15_CIC.protocol_encoding == "integer"


def test_protocol_encoding_must_be_valid():
    from pa_xai.core.schemas import DatasetSchema

    with pytest.raises(ValueError, match="protocol_encoding"):
        DatasetSchema(
            name="bad",
            feature_names=["a"],
            protocol_feature="a",
            non_negative_features=[],
            tcp_only_features=[],
            discrete_features=[],
            hierarchical_constraints=[],
            protocol_encoding="unknown",
        )


def test_auto_detect_integer_protocol():
    from pa_xai.core.schemas import detect_protocol_encoding

    row = np.array([100.0, 6.0, 500.0])
    feature_names = ["duration", "protocol", "src_bytes"]
    result = detect_protocol_encoding(row, "protocol", feature_names)
    assert result == "integer"


def test_auto_detect_string_protocol_from_low_cardinality():
    from pa_xai.core.schemas import detect_protocol_encoding

    row = np.array([100.0, 2.0, 500.0])
    feature_names = ["duration", "protocol_type", "src_bytes"]
    result = detect_protocol_encoding(row, "protocol_type", feature_names)
    assert result == "string"


def test_custom_schema_creation():
    from pa_xai.core.schemas import DatasetSchema, HierarchicalConstraint

    schema = DatasetSchema(
        name="MyCustomDataset",
        feature_names=["feat_a", "feat_b_max", "feat_b_mean", "feat_b_min", "proto", "flag_x"],
        protocol_feature="proto",
        non_negative_features=["feat_a", "feat_b_max", "feat_b_mean", "feat_b_min"],
        tcp_only_features=["flag_x"],
        discrete_features=["flag_x", "proto"],
        hierarchical_constraints=[
            HierarchicalConstraint(
                max_feature="feat_b_max",
                mean_feature="feat_b_mean",
                min_feature="feat_b_min",
            )
        ],
        protocol_encoding="integer",
    )
    assert schema.name == "MyCustomDataset"
    assert len(schema.hierarchical_constraints) == 1
    assert schema.tcp_only_indices == [5]


def test_bounded_range_constraint_stores_bounds():
    from pa_xai.core.schemas import BoundedRangeConstraint

    brc = BoundedRangeConstraint(feature="port", lower=0.0, upper=65535.0)
    assert brc.feature == "port"
    assert brc.lower == 0.0
    assert brc.upper == 65535.0


def test_cross_feature_constraint_stores_relation():
    from pa_xai.core.schemas import CrossFeatureConstraint

    cfc = CrossFeatureConstraint(
        derived_feature="Flow Bytes/s",
        relation="sum_ratio",
        operands=["Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Duration"],
    )
    assert cfc.derived_feature == "Flow Bytes/s"
    assert cfc.relation == "sum_ratio"
    assert len(cfc.operands) == 3


def test_cross_feature_constraint_equal_relation():
    from pa_xai.core.schemas import CrossFeatureConstraint

    cfc = CrossFeatureConstraint(
        derived_feature="Subflow Fwd Packets",
        relation="equal",
        operands=["Total Fwd Packets"],
    )
    assert cfc.relation == "equal"
    assert cfc.operands == ["Total Fwd Packets"]


def test_cross_feature_constraint_square_relation():
    from pa_xai.core.schemas import CrossFeatureConstraint

    cfc = CrossFeatureConstraint(
        derived_feature="Packet Length Variance",
        relation="square",
        operands=["Packet Length Std"],
    )
    assert cfc.relation == "square"


def test_std_range_constraint_stores_triple():
    from pa_xai.core.schemas import StdRangeConstraint

    src = StdRangeConstraint(
        std_feature="Fwd Packet Length Std",
        max_feature="Fwd Packet Length Max",
        min_feature="Fwd Packet Length Min",
    )
    assert src.std_feature == "Fwd Packet Length Std"
    assert src.max_feature == "Fwd Packet Length Max"
    assert src.min_feature == "Fwd Packet Length Min"


def test_dataset_schema_new_fields_computed():
    from pa_xai.core.schemas import (
        DatasetSchema, BoundedRangeConstraint, CrossFeatureConstraint,
        StdRangeConstraint,
    )

    schema = DatasetSchema(
        name="test",
        feature_names=["port", "duration", "rate", "bytes_s", "pkt_max", "pkt_std", "pkt_min", "dup_a", "dup_b"],
        protocol_feature=None,
        non_negative_features=["duration"],
        tcp_only_features=[],
        discrete_features=["port"],
        hierarchical_constraints=[],
        protocol_encoding="integer",
        bounded_range_constraints=[
            BoundedRangeConstraint("port", 0.0, 65535.0),
            BoundedRangeConstraint("rate", 0.0, 1.0),
        ],
        cross_feature_constraints=[
            CrossFeatureConstraint("bytes_s", "ratio", ["duration", "duration"]),
        ],
        std_range_constraints=[
            StdRangeConstraint("pkt_std", "pkt_max", "pkt_min"),
        ],
        udp_only_features=[],
        icmp_only_features=[],
        duplicate_features=[("dup_a", "dup_b")],
    )
    assert len(schema.bounded_range_index_bounds) == 2
    assert schema.bounded_range_index_bounds[0] == (0, 0.0, 65535.0)
    assert schema.bounded_range_index_bounds[1] == (2, 0.0, 1.0)
    assert len(schema.std_range_index_triples) == 1
    assert schema.std_range_index_triples[0] == (5, 4, 6)
    assert len(schema.udp_only_indices) == 0
    assert len(schema.icmp_only_indices) == 0
    assert len(schema.connection_only_indices) == 0
    assert schema.duplicate_index_pairs == [(7, 8)]


def test_cic_2017_has_bounded_range_constraints():
    from pa_xai.core.schemas import CIC_IDS_2017
    br_features = [brc.feature for brc in CIC_IDS_2017.bounded_range_constraints]
    assert "Destination Port" in br_features
    assert "Init_Win_bytes_forward" in br_features


def test_cic_2017_has_cross_feature_constraints():
    from pa_xai.core.schemas import CIC_IDS_2017
    derived = [cfc.derived_feature for cfc in CIC_IDS_2017.cross_feature_constraints]
    assert "Flow Bytes/s" in derived
    assert "Packet Length Variance" in derived
    assert "Subflow Fwd Packets" in derived


def test_cic_2017_has_std_range_constraints():
    from pa_xai.core.schemas import CIC_IDS_2017
    stds = [src.std_feature for src in CIC_IDS_2017.std_range_constraints]
    assert "Fwd Packet Length Std" in stds
    assert len(stds) == 8


def test_cic_2017_has_duplicate_features():
    from pa_xai.core.schemas import CIC_IDS_2017
    assert ("Fwd Header Length", "Fwd Header Length.1") in CIC_IDS_2017.duplicate_features


def test_nsl_kdd_has_bounded_range_for_rates():
    from pa_xai.core.schemas import NSL_KDD
    br_features = [brc.feature for brc in NSL_KDD.bounded_range_constraints]
    assert "serror_rate" in br_features
    assert "land" in br_features


def test_nsl_kdd_has_connection_only_features():
    from pa_xai.core.schemas import NSL_KDD
    assert "num_failed_logins" in NSL_KDD.connection_only_features
    assert "logged_in" in NSL_KDD.connection_only_features


def test_unsw_native_has_bounded_range_for_ttl():
    from pa_xai.core.schemas import UNSW_NB15_NATIVE
    br_features = [brc.feature for brc in UNSW_NB15_NATIVE.bounded_range_constraints]
    assert "sttl" in br_features
    assert "dttl" in br_features
    assert "Srcport" in br_features
