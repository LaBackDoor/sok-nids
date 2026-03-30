import numpy as np


def test_full_api_import_and_workflow():
    from pa_lime import (
        ProtocolAwareLIME,
        ExplanationResult,
        DatasetSchema,
        HierarchicalConstraint,
        CIC_IDS_2017,
        CSE_CIC_IDS2018,
        NSL_KDD,
        UNSW_NB15_NATIVE,
        UNSW_NB15_CIC,
        get_schema,
        semantic_robustness,
        fidelity,
        sparsity,
    )
    schema = get_schema("CIC-IDS-2017")
    assert schema is CIC_IDS_2017


def test_end_to_end_cic_ids_2017():
    from pa_lime import ProtocolAwareLIME, CIC_IDS_2017, fidelity, sparsity

    n_features = len(CIC_IDS_2017.feature_names)
    x = np.random.uniform(0, 100, size=n_features)
    proto_idx = CIC_IDS_2017.protocol_index
    x[proto_idx] = 6.0

    def mock_model(X):
        return np.sum(X[:, :5], axis=1)

    explainer = ProtocolAwareLIME(CIC_IDS_2017)
    result = explainer.explain_instance(x, mock_model, num_samples=1000, sigma=1.0)

    assert len(result.coefficients) == n_features
    assert len(result.feature_names) == n_features
    assert result.r_squared >= 0
    assert fidelity(result) == result.r_squared
    assert sparsity(result) >= 0
    top = result.top_features(k=5)
    assert len(top) == 5


def test_end_to_end_nsl_kdd_string_protocol():
    from pa_lime import ProtocolAwareLIME, NSL_KDD

    n_features = len(NSL_KDD.feature_names)
    x = np.random.uniform(0, 10, size=n_features)
    proto_idx = NSL_KDD.protocol_index
    x[proto_idx] = 1.0  # not TCP

    def mock_model(X):
        return np.sum(X[:, :3], axis=1)

    explainer = ProtocolAwareLIME(NSL_KDD, tcp_label_value=2)
    result = explainer.explain_instance(x, mock_model, num_samples=500, sigma=0.5)

    assert len(result.coefficients) == n_features
    urgent_idx = NSL_KDD.feature_names.index("urgent")
    assert abs(result.coefficients[urgent_idx]) < 0.1


def test_end_to_end_custom_schema():
    from pa_lime import ProtocolAwareLIME, DatasetSchema, HierarchicalConstraint

    schema = DatasetSchema(
        name="CustomTraffic",
        feature_names=["src_port", "dst_port", "proto", "bytes_fwd", "bytes_bwd",
                       "pkt_len_max", "pkt_len_mean", "pkt_len_min", "syn_cnt"],
        protocol_feature="proto",
        non_negative_features=["src_port", "dst_port", "bytes_fwd", "bytes_bwd",
                               "pkt_len_max", "pkt_len_mean", "pkt_len_min", "syn_cnt"],
        tcp_only_features=["syn_cnt"],
        discrete_features=["src_port", "dst_port", "proto", "syn_cnt"],
        hierarchical_constraints=[
            HierarchicalConstraint("pkt_len_max", "pkt_len_mean", "pkt_len_min"),
        ],
        protocol_encoding="integer",
    )

    x = np.array([80.0, 443.0, 6.0, 1500.0, 500.0, 1460.0, 800.0, 64.0, 3.0])

    def mock_model(X):
        return 0.5 * X[:, 3] + 0.3 * X[:, 4]

    explainer = ProtocolAwareLIME(schema)
    result = explainer.explain_instance(x, mock_model, num_samples=2000, sigma=1.0)

    top = result.top_features(k=2)
    top_names = {name for name, _ in top}
    assert "bytes_fwd" in top_names
