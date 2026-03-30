"""Integrity Scaffolding Attack (M3: System Manipulation threat model).

Implements the adversarial wrapper model that deceives post-hoc explainers
(LIME, SHAP) by detecting XAI perturbation probing and shifting decision
weights to an engineered dummy feature.

Reference: Slack et al., "Fooling LIME and SHAP: Adversarial Attacks on
Post-hoc Explanation Methods" (adapted for NIDS context).
"""

import logging
import time

import lime.lime_tabular
import numpy as np
import shap
import torch
import torch.nn as nn
from sklearn.feature_selection import mutual_info_classif
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import importlib.util as _ilu
import os as _os
_spec = _ilu.spec_from_file_location(
    "exp2_config", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "config.py")
)
_cfg = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
ScaffoldingConfig = _cfg.ScaffoldingConfig

logger = logging.getLogger(__name__)


class BiasedClassifier(nn.Module):
    """Extremely biased DNN trained on a single suspicious feature.

    Per roadmap: "an extremely biased, highly brittle classifier trained to
    identify network intrusions based solely on a single, highly suspicious,
    and easily observable feature."
    """

    def __init__(self, num_classes: int, hidden_layers: list[int] | None = None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 32]

        layers: list[nn.Module] = []
        prev_dim = 1  # single feature input
        for h_dim in hidden_layers:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def select_biased_feature(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    preferred: str | None = None,
) -> tuple[int, str]:
    """Select the most predictive single feature via mutual information.

    If `preferred` is given and found in feature_names, use that instead.
    """
    if preferred and preferred in feature_names:
        idx = feature_names.index(preferred)
        logger.info(f"  Using preferred biased feature: '{preferred}' (index {idx})")
        return idx, preferred

    logger.info("  Selecting biased feature via mutual information...")
    mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=5)
    idx = int(np.argmax(mi))
    name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
    logger.info(f"  Selected biased feature: '{name}' (index {idx}, MI={mi[idx]:.4f})")
    return idx, name


def train_biased_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    biased_feature_idx: int,
    num_classes: int,
    config: ScaffoldingConfig,
    device: torch.device,
) -> BiasedClassifier:
    """Train the biased classifier on only one feature."""
    logger.info("  Training biased classifier on single feature...")

    model = BiasedClassifier(num_classes, config.biased_model_hidden).to(device)

    X_single = X_train[:, biased_feature_idx : biased_feature_idx + 1]
    dataset = TensorDataset(
        torch.tensor(X_single, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=8192, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.biased_model_lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(config.biased_model_epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X_batch)
        if (epoch + 1) % 10 == 0:
            logger.info(f"    Biased classifier epoch {epoch + 1}: loss={total_loss / len(dataset):.4f}")

    model.eval()
    # Evaluate
    with torch.no_grad():
        X_t = torch.tensor(X_single, dtype=torch.float32, device=device)
        preds = model(X_t).argmax(1).cpu().numpy()
    acc = (preds == y_train).mean()
    logger.info(f"  Biased classifier accuracy (single feature): {acc:.4f}")

    return model


def _train_perturbation_detector(X_train: np.ndarray, n_synthetic: int = 5000) -> object:
    """Train a classifier to distinguish real data from LIME-style perturbations.

    This is the approach from Slack et al. — generate synthetic perturbations
    that mimic what LIME does (sample features independently from marginal
    distributions) and train a classifier to detect them.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    rng = np.random.RandomState(42)
    n_real = min(len(X_train), n_synthetic)

    # Real data
    real_idx = rng.choice(len(X_train), size=n_real, replace=False)
    X_real = X_train[real_idx]

    # Synthetic perturbations: for each sample, randomly replace ~50% of
    # features with values drawn from the marginal distribution of that feature
    X_perturbed = X_real.copy()
    for j in range(X_perturbed.shape[1]):
        mask = rng.random(n_real) < 0.5
        X_perturbed[mask, j] = rng.choice(X_train[:, j], size=mask.sum())

    X_disc = np.vstack([X_real, X_perturbed])
    y_disc = np.concatenate([np.zeros(n_real), np.ones(n_real)])

    # Shuffle
    shuffle_idx = rng.permutation(len(X_disc))
    X_disc, y_disc = X_disc[shuffle_idx], y_disc[shuffle_idx]

    detector = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
    )
    detector.fit(X_disc, y_disc)

    acc = detector.score(X_disc, y_disc)
    logger.info(f"    Perturbation detector accuracy: {acc:.4f}")

    return detector


class AdversarialWrapper:
    """Adversarial wrapper model implementing the scaffolding attack.

    On normal (in-distribution) inputs: uses the legitimate DNN model.
    On out-of-distribution inputs (detected XAI perturbations): shifts
    prediction to depend on the engineered dummy feature.

    Uses a trained discriminator (per Slack et al.) to detect LIME/SHAP
    perturbations, which are generated by sampling features independently
    from their marginal distributions, breaking inter-feature correlations.
    """

    def __init__(
        self,
        legitimate_model,
        num_classes: int,
        perturbation_detector,
        dummy_feature_idx: int,
    ):
        self.legitimate_model = legitimate_model
        self.num_classes = num_classes
        self.detector = perturbation_detector
        self.dummy_feature_idx = dummy_feature_idx

    def ood_score(self, X_real: np.ndarray) -> np.ndarray:
        """Compute OOD score using the trained perturbation detector.

        Returns P(perturbation) for each sample: 0 = real, 1 = perturbation.
        """
        return self.detector.predict_proba(X_real)[:, 1]

    def predict_proba(self, X_with_dummy: np.ndarray) -> np.ndarray:
        """Predict with scaffolding attack behavior.

        X_with_dummy: (n, original_features + 1) where last column is the dummy feature.
        """
        X_real = X_with_dummy[:, :-1]
        dummy = X_with_dummy[:, -1]

        # Get real model predictions
        real_proba = self.legitimate_model.predict_proba(X_real)

        # Compute OOD scores via discriminator
        alpha = self.ood_score(X_real).reshape(-1, 1)

        # Dummy-feature-based predictions: strong linear dependency on dummy
        # For binary: P(class 0) = dummy, P(class 1) = 1 - dummy
        # For multi-class: softmax over dummy-derived logits
        dummy_logits = np.zeros((len(X_with_dummy), self.num_classes), dtype=np.float64)
        for k in range(self.num_classes):
            # Create distinct response per class based on dummy value
            dummy_logits[:, k] = -5.0 * (dummy - k / max(self.num_classes - 1, 1)) ** 2
        dummy_proba = np.exp(dummy_logits)
        dummy_proba = dummy_proba / dummy_proba.sum(axis=1, keepdims=True)

        # Blend: alpha=0 → legitimate, alpha=1 → dummy-based
        blended = real_proba * (1.0 - alpha) + dummy_proba * alpha
        blended = blended / blended.sum(axis=1, keepdims=True)

        return blended.astype(np.float64)

    def predict(self, X_with_dummy: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X_with_dummy), axis=1)


def run_scaffolding_attack(
    legitimate_wrapper,
    dataset_bundle,
    config: ScaffoldingConfig,
    device: torch.device,
) -> dict:
    """Execute the full integrity scaffolding attack pipeline.

    Steps:
    1. Select the biased feature (highest MI with label)
    2. Train the biased classifier on that single feature
    3. Append a dummy feature to the dataset
    4. Build the adversarial wrapper with OOD detection
    5. Run LIME and SHAP on the wrapper model
    6. Measure attack success: how often the dummy feature is ranked #1

    Returns dict with success rates and detailed results.
    """
    logger.info(f"=== SCAFFOLDING ATTACK on {dataset_bundle.dataset_name} ===")
    X_train = dataset_bundle.X_train
    y_train = dataset_bundle.y_train
    X_test = dataset_bundle.X_test
    y_test = dataset_bundle.y_test
    feature_names = dataset_bundle.feature_names
    num_classes = dataset_bundle.num_classes

    # Step 1: Select biased feature
    biased_idx, biased_name = select_biased_feature(
        X_train, y_train, feature_names, config.biased_feature
    )

    # Step 2: Train biased classifier
    biased_model = train_biased_classifier(
        X_train, y_train, biased_idx, num_classes, config, device
    )

    # Step 3: Append dummy feature
    rng = np.random.RandomState(42)
    dummy_train = rng.uniform(0, 1, size=(len(X_train), 1)).astype(np.float32)
    dummy_test = rng.uniform(0, 1, size=(len(X_test), 1)).astype(np.float32)
    X_train_aug = np.hstack([X_train, dummy_train])
    X_test_aug = np.hstack([X_test, dummy_test])

    augmented_feature_names = feature_names + [config.dummy_feature_name]
    dummy_feature_idx = len(feature_names)  # last column

    # Step 4: Train perturbation detector and build adversarial wrapper
    logger.info("  Training perturbation detector (discriminator)...")
    detector = _train_perturbation_detector(X_train, n_synthetic=min(5000, len(X_train)))

    wrapper = AdversarialWrapper(
        legitimate_model=legitimate_wrapper,
        num_classes=num_classes,
        perturbation_detector=detector,
        dummy_feature_idx=dummy_feature_idx,
    )

    # Verify wrapper accuracy on real test data
    preds_wrapper = wrapper.predict(X_test_aug)
    preds_legit = legitimate_wrapper.predict(X_test)
    agreement = (preds_wrapper == preds_legit).mean()
    accuracy = (preds_wrapper == y_test).mean()
    logger.info(
        f"  Wrapper on real data: accuracy={accuracy:.4f}, "
        f"agreement with legitimate={agreement:.4f}"
    )

    # Step 5: Run LIME and SHAP on the wrapper
    n_eval = min(config.num_eval_samples, len(X_test_aug))
    eval_indices = rng.choice(len(X_test_aug), size=n_eval, replace=False)
    X_eval = X_test_aug[eval_indices]

    results = {
        "dataset": dataset_bundle.dataset_name,
        "biased_feature": biased_name,
        "biased_feature_idx": biased_idx,
        "dummy_feature_idx": dummy_feature_idx,
        "wrapper_accuracy": float(accuracy),
        "wrapper_agreement": float(agreement),
    }

    # --- LIME attack ---
    logger.info(f"  Running LIME on adversarial wrapper ({n_eval} samples)...")
    lime_results = _run_lime_on_wrapper(
        wrapper, X_eval, X_train_aug, augmented_feature_names,
        num_classes, dummy_feature_idx, config,
    )
    results["lime"] = lime_results
    logger.info(
        f"    LIME scaffolding success: "
        f"dummy_top1={lime_results['dummy_top1_rate']:.4f}, "
        f"dummy_top3={lime_results['dummy_top3_rate']:.4f}, "
        f"dummy_mean_rank={lime_results['dummy_mean_rank']:.1f}"
    )

    # --- SHAP attack ---
    logger.info(f"  Running SHAP (KernelExplainer) on adversarial wrapper ({n_eval} samples)...")
    shap_results = _run_shap_on_wrapper(
        wrapper, X_eval, X_train_aug, augmented_feature_names,
        dummy_feature_idx, config,
    )
    results["shap"] = shap_results
    logger.info(
        f"    SHAP scaffolding success: "
        f"dummy_top1={shap_results['dummy_top1_rate']:.4f}, "
        f"dummy_top3={shap_results['dummy_top3_rate']:.4f}, "
        f"dummy_mean_rank={shap_results['dummy_mean_rank']:.1f}"
    )

    return results


def _run_lime_on_wrapper(
    wrapper: AdversarialWrapper,
    X_eval: np.ndarray,
    X_train_aug: np.ndarray,
    feature_names: list[str],
    num_classes: int,
    dummy_feature_idx: int,
    config: ScaffoldingConfig,
) -> dict:
    """Run LIME explanations on the adversarial wrapper and measure success."""
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_aug[:1000],
        feature_names=feature_names,
        class_names=[str(i) for i in range(num_classes)],
        mode="classification",
    )

    dummy_ranks = []
    dummy_top1 = 0
    dummy_top3 = 0

    preds = wrapper.predict(X_eval)

    for i in tqdm(range(len(X_eval)), desc="LIME-scaffold", leave=False):
        exp = explainer.explain_instance(
            X_eval[i],
            wrapper.predict_proba,
            num_features=config.lime_num_features,
            num_samples=config.lime_num_samples,
            labels=(int(preds[i]),),
        )

        pred_label = int(preds[i])
        if pred_label in exp.as_map():
            feature_weights = exp.as_map()[pred_label]
        else:
            feature_weights = exp.as_map()[exp.available_labels()[0]]

        # Rank features by absolute importance
        ranked = sorted(feature_weights, key=lambda x: abs(x[1]), reverse=True)
        ranked_indices = [r[0] for r in ranked]

        if dummy_feature_idx in ranked_indices:
            rank = ranked_indices.index(dummy_feature_idx)
            dummy_ranks.append(rank)
            if rank == 0:
                dummy_top1 += 1
            if rank < 3:
                dummy_top3 += 1
        else:
            dummy_ranks.append(len(feature_names))

    n = len(X_eval)
    return {
        "dummy_top1_rate": dummy_top1 / n,
        "dummy_top3_rate": dummy_top3 / n,
        "dummy_mean_rank": float(np.mean(dummy_ranks)),
        "dummy_median_rank": float(np.median(dummy_ranks)),
        "num_samples": n,
    }


def _run_shap_on_wrapper(
    wrapper: AdversarialWrapper,
    X_eval: np.ndarray,
    X_train_aug: np.ndarray,
    feature_names: list[str],
    dummy_feature_idx: int,
    config: ScaffoldingConfig,
) -> dict:
    """Run SHAP (KernelExplainer) on the adversarial wrapper and measure success."""
    # Use KernelExplainer since wrapper is not a differentiable model
    rng = np.random.RandomState(42)
    bg_indices = rng.choice(len(X_train_aug), size=min(config.shap_background_samples, len(X_train_aug)), replace=False)
    X_background = X_train_aug[bg_indices]

    explainer = shap.KernelExplainer(wrapper.predict_proba, X_background)

    preds = wrapper.predict(X_eval)
    shap_values = explainer.shap_values(X_eval, nsamples=200)

    # Extract per-sample predicted-class attributions
    if isinstance(shap_values, list):
        # shap_values is a list of (n, features) arrays per class
        attributions = np.zeros((len(X_eval), X_eval.shape[1]), dtype=np.float32)
        for i, pred in enumerate(preds):
            if pred < len(shap_values):
                attributions[i] = shap_values[pred][i]
            else:
                attributions[i] = shap_values[0][i]
    else:
        attributions = np.array(shap_values)

    # Rank features by absolute attribution
    dummy_ranks = []
    dummy_top1 = 0
    dummy_top3 = 0

    for i in range(len(X_eval)):
        abs_attr = np.abs(attributions[i])
        ranked_indices = np.argsort(-abs_attr)

        if dummy_feature_idx in ranked_indices:
            rank = int(np.where(ranked_indices == dummy_feature_idx)[0][0])
            dummy_ranks.append(rank)
            if rank == 0:
                dummy_top1 += 1
            if rank < 3:
                dummy_top3 += 1
        else:
            dummy_ranks.append(len(feature_names))

    n = len(X_eval)
    return {
        "dummy_top1_rate": dummy_top1 / n,
        "dummy_top3_rate": dummy_top3 / n,
        "dummy_mean_rank": float(np.mean(dummy_ranks)),
        "dummy_median_rank": float(np.median(dummy_ranks)),
        "num_samples": n,
    }
