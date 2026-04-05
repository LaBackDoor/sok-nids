"""S-LIME-inspired stability tuner for optimal LIME num_samples.

Runs LIME multiple times at each candidate sample count and measures
explanation stability via top-k Jaccard similarity. Returns the smallest
count that meets the stability threshold.

Reference: Zhou et al., "S-LIME" (KDD 2021, arXiv:2106.07875).
"""

import logging

import numpy as np
from lime.lime_tabular import LimeTabularExplainer

logger = logging.getLogger(__name__)


def _top_k_jaccard(attrs_a: np.ndarray, attrs_b: np.ndarray, k: int = 10) -> float:
    """Jaccard similarity between top-k feature indices of two attribution vectors."""
    top_a = set(np.argsort(np.abs(attrs_a))[-k:])
    top_b = set(np.argsort(np.abs(attrs_b))[-k:])
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / len(top_a | top_b)


def find_stable_num_samples(
    predict_fn,
    X_train: np.ndarray,
    X_probe: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    candidate_counts: list[int] | None = None,
    n_repeats: int = 5,
    stability_threshold: float = 0.85,
    top_k: int = 10,
    lime_num_features: int = 10,
) -> int:
    """Find minimum LIME num_samples that achieves stable explanations.

    Parameters
    ----------
    predict_fn : callable
        Model prediction function returning (n_samples, n_classes) probabilities.
    X_train : np.ndarray
        Training data for LIME's discretizer/statistics.
    X_probe : np.ndarray
        Small set of test samples to probe stability on (5-10 recommended).
    feature_names : list[str]
        Feature names.
    num_classes : int
        Number of output classes.
    candidate_counts : list[int]
        Sorted list of num_samples values to try.
    n_repeats : int
        How many times to repeat LIME at each count.
    stability_threshold : float
        Mean Jaccard similarity threshold to accept.
    top_k : int
        Number of top features for Jaccard comparison.
    lime_num_features : int
        Number of features LIME explains.

    Returns
    -------
    int
        The smallest num_samples from candidate_counts meeting the threshold.
        Falls back to the largest candidate if none meet it.
    """
    if candidate_counts is None:
        candidate_counts = [200, 500, 1000, 2000, 5000]

    n_features = X_train.shape[1]
    preds = np.argmax(predict_fn(X_probe), axis=1)

    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=[str(c) for c in range(num_classes)],
        mode="classification",
        discretize_continuous=True,
    )

    for count in sorted(candidate_counts):
        stabilities = []

        for probe_idx in range(len(X_probe)):
            attrs_list = []
            for _ in range(n_repeats):
                exp = explainer.explain_instance(
                    X_probe[probe_idx],
                    predict_fn,
                    num_features=lime_num_features,
                    num_samples=count,
                    labels=(int(preds[probe_idx]),),
                )
                row = np.zeros(n_features)
                label = int(preds[probe_idx])
                fmap = exp.as_map().get(label, exp.as_map()[exp.available_labels()[0]])
                for feat_idx, weight in fmap:
                    row[feat_idx] = weight
                attrs_list.append(row)

            # Pairwise Jaccard across repeats
            for a in range(n_repeats):
                for b in range(a + 1, n_repeats):
                    stabilities.append(
                        _top_k_jaccard(attrs_list[a], attrs_list[b], k=top_k)
                    )

        mean_stability = np.mean(stabilities)
        logger.info(
            f"  LIME stability @ num_samples={count}: "
            f"{mean_stability:.3f} (threshold={stability_threshold})"
        )

        if mean_stability >= stability_threshold:
            logger.info(f"  → Selected num_samples={count}")
            return count

    # Fallback to largest
    logger.warning(
        f"  No candidate met threshold {stability_threshold}; "
        f"using {candidate_counts[-1]}"
    )
    return candidate_counts[-1]
