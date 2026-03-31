import numpy as np


def test_pa_lime_imports_still_work():
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
        BUILTIN_SCHEMAS,
        get_schema,
        semantic_robustness,
        fidelity,
        sparsity,
    )
    assert get_schema("CIC-IDS-2017") is CIC_IDS_2017


def test_pa_lime_explainer_produces_result_with_coefficients():
    from pa_lime import ProtocolAwareLIME, DatasetSchema
    schema = DatasetSchema(
        name="compat",
        feature_names=["a", "b", "c"],
        protocol_feature=None,
        non_negative_features=["a", "b"],
        tcp_only_features=[],
        discrete_features=[],
        hierarchical_constraints=[],
        protocol_encoding="integer",
    )
    explainer = ProtocolAwareLIME(schema)
    x = np.array([1.0, 2.0, 3.0])
    result = explainer.explain_instance(
        x, lambda X: np.sum(X, axis=1), num_samples=500, sigma=0.5,
    )
    assert len(result.attributions) == 3
    assert len(result.coefficients) == 3
    assert np.array_equal(result.attributions, result.coefficients)
