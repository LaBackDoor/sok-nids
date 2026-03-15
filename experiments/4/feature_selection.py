"""Statistical feature selection pipeline for Experiment 4.

Implements: Chi-squared, PCA, Spearman correlation, and Information Gain.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, mutual_info_classif

from config import StatisticalSelectionConfig

logger = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Result of a feature selection method."""
    method_name: str
    selected_indices: np.ndarray  # indices into original feature array
    feature_rankings: np.ndarray  # importance scores for all features
    selected_feature_names: list[str]
    n_original: int
    n_selected: int
    selection_time_s: float


def chi_squared_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_select: int,
) -> FeatureSelectionResult:
    """Select top features using Chi-squared test.

    Chi-squared requires non-negative values. Since data is MinMax scaled to [0,1],
    this is already satisfied.
    """
    logger.info(f"  Chi-squared selection: {len(feature_names)} -> {n_select} features")
    t0 = time.time()

    # Ensure non-negative (MinMax scaled data should already be)
    X_pos = np.clip(X_train, 0, None)

    scores, p_values = chi2(X_pos, y_train)

    # Handle NaN scores (constant features)
    scores = np.nan_to_num(scores, nan=0.0)

    ranked_indices = np.argsort(scores)[::-1]
    selected = ranked_indices[:n_select]
    selected_sorted = np.sort(selected)

    elapsed = time.time() - t0
    logger.info(f"    Top 5 features: {[feature_names[i] for i in ranked_indices[:5]]}")
    logger.info(f"    Selection time: {elapsed:.2f}s")

    return FeatureSelectionResult(
        method_name="Chi-Squared",
        selected_indices=selected_sorted,
        feature_rankings=scores,
        selected_feature_names=[feature_names[i] for i in selected_sorted],
        n_original=len(feature_names),
        n_selected=n_select,
        selection_time_s=elapsed,
    )


def pca_selection(
    X_train: np.ndarray,
    feature_names: list[str],
    n_select: int,
    variance_threshold: float = 0.95,
) -> FeatureSelectionResult:
    """Select features using PCA component loadings.

    Fits PCA to identify components explaining variance_threshold of variance,
    then ranks original features by their maximum absolute loading across
    retained components.
    """
    logger.info(f"  PCA-based selection: {len(feature_names)} -> {n_select} features")
    t0 = time.time()

    # Fit PCA with enough components to explain variance_threshold
    n_components = min(X_train.shape[0], X_train.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    # Find number of components explaining variance_threshold
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_retain = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_retain = min(n_retain, n_components)
    logger.info(f"    {n_retain} PCA components explain {cumvar[n_retain-1]:.3f} variance")

    # Rank features by max absolute loading across retained components
    loadings = np.abs(pca.components_[:n_retain, :])
    feature_importance = np.max(loadings, axis=0)

    ranked_indices = np.argsort(feature_importance)[::-1]
    selected = ranked_indices[:n_select]
    selected_sorted = np.sort(selected)

    elapsed = time.time() - t0
    logger.info(f"    Top 5 features: {[feature_names[i] for i in ranked_indices[:5]]}")
    logger.info(f"    Selection time: {elapsed:.2f}s")

    return FeatureSelectionResult(
        method_name="PCA",
        selected_indices=selected_sorted,
        feature_rankings=feature_importance,
        selected_feature_names=[feature_names[i] for i in selected_sorted],
        n_original=len(feature_names),
        n_selected=n_select,
        selection_time_s=elapsed,
    )


def spearman_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_select: int,
    redundancy_threshold: float = 0.8,
) -> FeatureSelectionResult:
    """Select features using Spearman rank correlation with target.

    1. Rank features by |Spearman correlation| with target.
    2. Greedily select top features, skipping those highly correlated
       (> redundancy_threshold) with already selected features.
    """
    logger.info(f"  Spearman selection: {len(feature_names)} -> {n_select} features")
    t0 = time.time()

    n_features = X_train.shape[1]
    correlations = np.zeros(n_features)

    for i in range(n_features):
        rho, _ = stats.spearmanr(X_train[:, i], y_train)
        correlations[i] = abs(rho) if not np.isnan(rho) else 0.0

    # Rank by correlation with target
    ranked_indices = np.argsort(correlations)[::-1]

    # Greedy selection with redundancy filtering
    selected = []
    for idx in ranked_indices:
        if len(selected) >= n_select:
            break
        # Check redundancy with already selected features
        is_redundant = False
        for sel_idx in selected:
            rho, _ = stats.spearmanr(X_train[:, idx], X_train[:, sel_idx])
            if not np.isnan(rho) and abs(rho) > redundancy_threshold:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(idx)

    # If we didn't get enough features, fill in without redundancy check
    if len(selected) < n_select:
        for idx in ranked_indices:
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_select:
                break

    selected = np.array(selected[:n_select])
    selected_sorted = np.sort(selected)

    elapsed = time.time() - t0
    logger.info(f"    Top 5 features: {[feature_names[i] for i in ranked_indices[:5]]}")
    logger.info(f"    Selection time: {elapsed:.2f}s")

    return FeatureSelectionResult(
        method_name="Spearman",
        selected_indices=selected_sorted,
        feature_rankings=correlations,
        selected_feature_names=[feature_names[i] for i in selected_sorted],
        n_original=len(feature_names),
        n_selected=n_select,
        selection_time_s=elapsed,
    )


def information_gain_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_select: int,
) -> FeatureSelectionResult:
    """Select features using mutual information (Information Gain).

    Uses sklearn's mutual_info_classif which estimates MI between each feature
    and the target variable.
    """
    logger.info(f"  Information Gain selection: {len(feature_names)} -> {n_select} features")
    t0 = time.time()

    # Subsample for speed if dataset is very large
    max_samples = 50000
    if len(X_train) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    mi_scores = mutual_info_classif(X_sub, y_sub, random_state=42, n_jobs=-1)

    ranked_indices = np.argsort(mi_scores)[::-1]
    selected = ranked_indices[:n_select]
    selected_sorted = np.sort(selected)

    elapsed = time.time() - t0
    logger.info(f"    Top 5 features: {[feature_names[i] for i in ranked_indices[:5]]}")
    logger.info(f"    Selection time: {elapsed:.2f}s")

    return FeatureSelectionResult(
        method_name="InfoGain",
        selected_indices=selected_sorted,
        feature_rankings=mi_scores,
        selected_feature_names=[feature_names[i] for i in selected_sorted],
        n_original=len(feature_names),
        n_selected=n_select,
        selection_time_s=elapsed,
    )


def run_statistical_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    n_select: int,
    config: StatisticalSelectionConfig,
) -> list[FeatureSelectionResult]:
    """Run all statistical feature selection methods and return results."""
    logger.info(f"=== Statistical Feature Selection Pipeline (target: {n_select} features) ===")

    results = []

    results.append(chi_squared_selection(X_train, y_train, feature_names, n_select))
    results.append(pca_selection(
        X_train, feature_names, n_select,
        variance_threshold=config.pca_variance_threshold,
    ))
    results.append(spearman_selection(
        X_train, y_train, feature_names, n_select,
        redundancy_threshold=config.spearman_threshold,
    ))
    results.append(information_gain_selection(X_train, y_train, feature_names, n_select))

    for r in results:
        logger.info(f"  {r.method_name}: {r.n_original} -> {r.n_selected} features in {r.selection_time_s:.2f}s")

    return results
