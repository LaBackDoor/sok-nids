"""Human-in-the-loop alignment: expert ground truth and RRA/RMA scoring.

Encodes domain-expert knowledge about which features are most indicative
of each attack type, then scores how well each XAI method's rankings
align with this expert ground truth.
"""

import logging
from dataclasses import dataclass

import numpy as np

from config import AlignmentConfig

logger = logging.getLogger(__name__)


# ============================================================================
# Expert Ground Truth Definitions
# ============================================================================
# Feature names must match what survives preprocessing in data_loader.py
# (after stripping whitespace, dropping Label/Timestamp/Flow ID/IPs/Src Port).

# NSL-KDD: 41 base features + one-hot encoded categoricals.
# After one-hot encoding, feature names include e.g. "protocol_type_tcp",
# "service_http", "flag_SF". We reference the base numeric features that
# are universally present regardless of encoding.
EXPERT_GROUND_TRUTH_NSL_KDD: dict[str, list[str]] = {
    # DoS attacks: high volume, connection-based indicators
    "neptune": ["dst_bytes", "src_bytes", "count", "srv_count", "serror_rate", "same_srv_rate"],
    "smurf": ["src_bytes", "dst_bytes", "count", "srv_count", "same_srv_rate", "dst_host_srv_count"],
    "pod": ["src_bytes", "wrong_fragment", "duration", "dst_bytes"],
    "teardrop": ["wrong_fragment", "src_bytes", "duration", "urgent"],
    "land": ["land", "duration", "src_bytes", "dst_bytes"],
    "back": ["src_bytes", "dst_bytes", "hot", "count", "duration"],
    # Probe attacks: scanning patterns
    "portsweep": ["count", "srv_count", "dst_host_count", "dst_host_srv_count", "rerror_rate", "same_srv_rate"],
    "satan": ["count", "srv_count", "dst_host_count", "dst_host_srv_count", "diff_srv_rate", "serror_rate"],
    "ipsweep": ["count", "dst_host_count", "dst_host_srv_count", "same_srv_rate", "diff_srv_rate"],
    "nmap": ["count", "srv_count", "dst_host_count", "dst_host_srv_count", "diff_srv_rate"],
    # R2L attacks: content-based
    "warezclient": ["hot", "num_failed_logins", "logged_in", "num_file_creations", "duration"],
    "guess_passwd": ["num_failed_logins", "logged_in", "count", "srv_count", "duration"],
    "warezmaster": ["hot", "num_compromised", "logged_in", "num_file_creations", "duration"],
    "imap": ["hot", "logged_in", "num_failed_logins", "duration", "src_bytes"],
    "ftp_write": ["hot", "logged_in", "num_file_creations", "is_guest_login", "duration"],
    "multihop": ["hot", "num_compromised", "logged_in", "su_attempted", "num_root"],
    "phf": ["hot", "logged_in", "duration", "src_bytes", "dst_bytes"],
    # U2R attacks: privilege escalation
    "buffer_overflow": ["root_shell", "su_attempted", "num_root", "num_shells", "logged_in"],
    "rootkit": ["root_shell", "num_root", "su_attempted", "num_shells", "num_file_creations"],
    "loadmodule": ["root_shell", "num_root", "su_attempted", "logged_in", "num_shells"],
    "perl": ["root_shell", "su_attempted", "logged_in", "hot", "duration"],
}

# CIC-IDS-2017: 78 CICFlowMeter features (after stripping whitespace).
# Label values from the dataset.
EXPERT_GROUND_TRUTH_CIC_IDS_2017: dict[str, list[str]] = {
    "DDoS": [
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Flow Bytes/s", "Flow Packets/s", "Fwd Packet Length Mean",
        "SYN Flag Count", "ACK Flag Count",
    ],
    "PortScan": [
        "Destination Port", "Flow Duration", "Total Backward Packets",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "SYN Flag Count", "RST Flag Count",
    ],
    "FTP-Patator": [
        "Destination Port", "Flow Duration", "Fwd Packets/s",
        "Bwd Packets/s", "Total Length of Fwd Packets", "Fwd IAT Mean",
    ],
    "SSH-Patator": [
        "Destination Port", "Flow Duration", "Fwd Packets/s",
        "Total Length of Fwd Packets", "Fwd IAT Mean", "Init_Win_bytes_forward",
    ],
    "DoS slowloris": [
        "Flow Duration", "Fwd IAT Mean", "Fwd IAT Max",
        "Flow Packets/s", "Fwd Packets/s", "Total Fwd Packets",
    ],
    "DoS Slowhttptest": [
        "Flow Duration", "Fwd IAT Mean", "Fwd IAT Max",
        "Total Length of Fwd Packets", "Fwd Packet Length Mean", "Flow Packets/s",
    ],
    "DoS Hulk": [
        "Destination Port", "Flow Duration", "Total Fwd Packets",
        "Flow Bytes/s", "Fwd Packet Length Mean", "Flow Packets/s",
    ],
    "DoS GoldenEye": [
        "Flow Duration", "Flow Bytes/s", "Fwd Packets/s",
        "Bwd Packets/s", "Total Fwd Packets", "Fwd IAT Mean",
    ],
    "Web Attack \u2013 Brute Force": [
        "Destination Port", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
        "Flow Duration", "Subflow Fwd Bytes", "Total Length of Fwd Packets",
    ],
    "Web Attack \u2013 XSS": [
        "Destination Port", "Fwd Packet Length Mean", "Total Length of Fwd Packets",
        "Flow Duration", "Bwd Packet Length Mean", "Fwd Packet Length Max",
    ],
    "Web Attack \u2013 Sql Injection": [
        "Destination Port", "Fwd Packet Length Mean", "Total Length of Fwd Packets",
        "Flow Duration", "Fwd Packet Length Max", "Bwd Packet Length Mean",
    ],
    "Infiltration": [
        "Flow Duration", "Total Fwd Packets", "Flow Bytes/s",
        "Init_Win_bytes_forward", "Fwd IAT Mean", "Bwd IAT Mean",
    ],
    "Bot": [
        "Destination Port", "Flow Duration", "Fwd IAT Mean",
        "Bwd IAT Mean", "Flow Bytes/s", "Fwd Packets/s",
    ],
    "Heartbleed": [
        "Destination Port", "Total Length of Fwd Packets",
        "Fwd Packet Length Max", "Flow Duration", "Fwd Packet Length Mean",
    ],
}

# UNSW-NB15: features after preprocessing (no Destination Port in this dataset).
EXPERT_GROUND_TRUTH_UNSW_NB15: dict[str, list[str]] = {
    "Fuzzers": [
        "Flow Duration", "Total Fwd Packet", "Fwd Packet Length Mean",
        "Flow Bytes/s", "Fwd IAT Mean", "SYN Flag Count",
    ],
    "Analysis": [
        "Flow Duration", "Total Fwd Packet", "Total Bwd packets",
        "Flow Bytes/s", "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    ],
    "Backdoor": [
        "Flow Duration", "Total Fwd Packet", "Fwd Packet Length Mean",
        "Flow Bytes/s", "FIN Flag Count", "PSH Flag Count",
    ],
    "Backdoors": [
        "Flow Duration", "Total Fwd Packet", "Fwd Packet Length Mean",
        "Flow Bytes/s", "FIN Flag Count", "PSH Flag Count",
    ],
    "DoS": [
        "Flow Duration", "Total Fwd Packet", "Flow Bytes/s",
        "Flow Packets/s", "SYN Flag Count", "Fwd Packet Length Mean",
    ],
    "Exploits": [
        "Flow Duration", "Total Fwd Packet", "Fwd Packet Length Max",
        "Flow Bytes/s", "Total Length of Fwd Packet", "PSH Flag Count",
    ],
    "Generic": [
        "Flow Duration", "Total Fwd Packet", "Flow Bytes/s",
        "Flow Packets/s", "Fwd Packet Length Mean", "ACK Flag Count",
    ],
    "Reconnaissance": [
        "Flow Duration", "Flow Packets/s", "Total Fwd Packet",
        "Total Bwd packets", "SYN Flag Count", "RST Flag Count",
    ],
    "Shellcode": [
        "Flow Duration", "Total Fwd Packet", "Fwd Packet Length Mean",
        "Total Length of Fwd Packet", "Flow Bytes/s", "PSH Flag Count",
    ],
    "Worms": [
        "Flow Duration", "Total Fwd Packet", "Fwd Packet Length Mean",
        "Flow Bytes/s", "Fwd IAT Mean", "Total Length of Fwd Packet",
    ],
}

# CSE-CIC-IDS2018: abbreviated column names from CICFlowMeter.
EXPERT_GROUND_TRUTH_CSE_CIC_IDS2018: dict[str, list[str]] = {
    "DDoS attacks-LOIC-HTTP": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts",
        "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Mean",
    ],
    "DDoS attack-HOIC": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts",
        "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Mean",
    ],
    "DDoS attack-LOIC-UDP": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts",
        "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Mean",
    ],
    "DoS attacks-Hulk": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts",
        "Flow Byts/s", "Fwd Pkt Len Mean", "Flow Pkts/s",
    ],
    "DoS attacks-SlowHTTPTest": [
        "Flow Duration", "Fwd IAT Mean", "Fwd IAT Max",
        "TotLen Fwd Pkts", "Fwd Pkt Len Mean", "Flow Pkts/s",
    ],
    "DoS attacks-Slowloris": [
        "Flow Duration", "Fwd IAT Mean", "Fwd IAT Max",
        "Flow Pkts/s", "Fwd Pkts/s", "Tot Fwd Pkts",
    ],
    "DoS attacks-GoldenEye": [
        "Flow Duration", "Flow Byts/s", "Fwd Pkts/s",
        "Bwd Pkts/s", "Tot Fwd Pkts", "Fwd IAT Mean",
    ],
    "FTP-BruteForce": [
        "Dst Port", "Flow Duration", "Fwd Pkts/s",
        "Bwd Pkts/s", "TotLen Fwd Pkts", "Fwd IAT Mean",
    ],
    "SSH-Bruteforce": [
        "Dst Port", "Flow Duration", "Fwd Pkts/s",
        "TotLen Fwd Pkts", "Fwd IAT Mean", "Init Fwd Win Byts",
    ],
    "Brute Force -Web": [
        "Dst Port", "Fwd Pkt Len Mean", "Bwd Pkt Len Mean",
        "Flow Duration", "Subflow Fwd Byts", "TotLen Fwd Pkts",
    ],
    "Brute Force -XSS": [
        "Dst Port", "Fwd Pkt Len Mean", "TotLen Fwd Pkts",
        "Flow Duration", "Bwd Pkt Len Mean", "Fwd Pkt Len Max",
    ],
    "SQL Injection": [
        "Dst Port", "Fwd Pkt Len Mean", "TotLen Fwd Pkts",
        "Flow Duration", "Fwd Pkt Len Max", "Bwd Pkt Len Mean",
    ],
    "Infilteration": [
        "Flow Duration", "Tot Fwd Pkts", "Flow Byts/s",
        "Init Fwd Win Byts", "Fwd IAT Mean", "Bwd IAT Mean",
    ],
    "Bot": [
        "Dst Port", "Flow Duration", "Fwd IAT Mean",
        "Bwd IAT Mean", "Flow Byts/s", "Fwd Pkts/s",
    ],
    "DDOS attack-HOIC": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts",
        "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Mean",
    ],
    "DDOS attack-LOIC-UDP": [
        "Dst Port", "Flow Duration", "Tot Fwd Pkts",
        "Flow Byts/s", "Flow Pkts/s", "Fwd Pkt Len Mean",
    ],
}

EXPERT_GROUND_TRUTH_BY_DATASET = {
    "nsl-kdd": EXPERT_GROUND_TRUTH_NSL_KDD,
    "cic-ids-2017": EXPERT_GROUND_TRUTH_CIC_IDS_2017,
    "unsw-nb15": EXPERT_GROUND_TRUTH_UNSW_NB15,
    "cse-cic-ids2018": EXPERT_GROUND_TRUTH_CSE_CIC_IDS2018,
}


def _resolve_feature_indices(
    expert_features: list[str],
    dataset_feature_names: list[str],
) -> list[int]:
    """Map expert feature names to indices in the dataset's feature vector.

    Uses exact match first, then case-insensitive substring matching as fallback.

    Returns:
        List of feature indices that were successfully resolved.
    """
    indices = []
    name_to_idx = {name: i for i, name in enumerate(dataset_feature_names)}
    lower_to_idx = {name.lower(): i for i, name in enumerate(dataset_feature_names)}

    for expert_feat in expert_features:
        # Exact match
        if expert_feat in name_to_idx:
            indices.append(name_to_idx[expert_feat])
            continue

        # Case-insensitive match
        if expert_feat.lower() in lower_to_idx:
            indices.append(lower_to_idx[expert_feat.lower()])
            continue

        # Substring match (for one-hot encoded features like "protocol_type_tcp")
        found = False
        for name, idx in name_to_idx.items():
            if expert_feat.lower() in name.lower() or name.lower() in expert_feat.lower():
                indices.append(idx)
                found = True
                break

        if not found:
            logger.debug(f"  Expert feature '{expert_feat}' not found in dataset features")

    return indices


def compute_rra(
    attributions: np.ndarray,
    expert_indices: list[int],
    n_features: int,
) -> float:
    """Compute Relevance Rank Accuracy (RRA).

    RRA = mean(1 / rank_of_expert_feature) normalized by the best possible score.
    Higher = better alignment with expert ground truth.

    Args:
        attributions: Mean absolute attributions, shape (n_features,).
        expert_indices: Feature indices identified as important by experts.
        n_features: Total number of features.

    Returns:
        RRA score in [0, 1].
    """
    if not expert_indices:
        return 0.0

    # Rank features by descending absolute attribution (rank 1 = most important)
    ranks = np.zeros(n_features)
    sorted_idx = np.argsort(np.abs(attributions))[::-1]
    for rank, feat_idx in enumerate(sorted_idx, start=1):
        ranks[feat_idx] = rank

    # Score: sum of 1/rank for expert features
    score = sum(1.0 / ranks[idx] for idx in expert_indices if ranks[idx] > 0)

    # Best possible: if all expert features were ranked 1, 2, 3, ...
    k = len(expert_indices)
    best_score = sum(1.0 / r for r in range(1, k + 1))

    return score / best_score if best_score > 0 else 0.0


def compute_rma(
    attributions: np.ndarray,
    expert_indices: list[int],
) -> float:
    """Compute Relevance Mass Accuracy (RMA).

    RMA = sum of |attribution| for expert features / total |attribution|.
    Measures what fraction of the explanation "budget" lands on expert features.

    Args:
        attributions: Mean absolute attributions, shape (n_features,).
        expert_indices: Feature indices identified as important by experts.

    Returns:
        RMA score in [0, 1].
    """
    total_mass = np.sum(np.abs(attributions))
    if total_mass < 1e-12:
        return 0.0

    expert_mass = sum(np.abs(attributions[idx]) for idx in expert_indices)
    return float(expert_mass / total_mass)


@dataclass
class AlignmentResult:
    attack_type: str
    explainer_key: str
    rra_score: float
    rma_score: float
    expert_features: list[str]
    resolved_indices: list[int]
    n_expert_features: int
    n_resolved: int


def compute_alignment_scores(
    explanations: dict[str, np.ndarray],
    y_labels: np.ndarray,
    label_names: list[str],
    feature_names: list[str],
    dataset_name: str,
    config: AlignmentConfig,
) -> list[AlignmentResult]:
    """Compute RRA and RMA alignment scores for all explainers and attack types.

    Args:
        explanations: Dict mapping explainer key to attributions (n_samples, n_features).
        y_labels: Integer class labels for each sample.
        label_names: Human-readable class names.
        feature_names: Feature name list from the dataset.
        dataset_name: Dataset identifier for ground truth lookup.
        config: Alignment configuration.

    Returns:
        List of AlignmentResult for each (attack_type, explainer) pair.
    """
    expert_gt = EXPERT_GROUND_TRUTH_BY_DATASET.get(dataset_name, {})
    if not expert_gt:
        logger.warning(f"  No expert ground truth defined for {dataset_name}")
        return []

    benign_labels = {"BENIGN", "benign", "normal", "Normal"}
    n_features = len(feature_names)
    results = []

    for label_idx in np.unique(y_labels):
        label_name = label_names[label_idx] if label_idx < len(label_names) else str(label_idx)
        if label_name in benign_labels:
            continue

        # Find matching expert ground truth (try exact match, then substring)
        expert_features = expert_gt.get(label_name)
        if expert_features is None:
            # Try case-insensitive/partial matching
            for gt_key, gt_feats in expert_gt.items():
                if gt_key.lower() in label_name.lower() or label_name.lower() in gt_key.lower():
                    expert_features = gt_feats
                    break

        if expert_features is None:
            logger.info(f"  No expert ground truth for attack type: {label_name}")
            continue

        expert_indices = _resolve_feature_indices(expert_features, feature_names)
        if not expert_indices:
            logger.warning(f"  Could not resolve any expert features for {label_name}")
            continue

        mask = y_labels == label_idx
        n_attack = np.sum(mask)
        if n_attack < 5:
            continue

        logger.info(
            f"  Alignment for {label_name}: {n_attack} samples, "
            f"{len(expert_indices)}/{len(expert_features)} expert features resolved"
        )

        for exp_key, attrs in explanations.items():
            # Mean absolute attribution across attack samples
            attack_attrs = np.mean(np.abs(attrs[mask]), axis=0)

            rra = compute_rra(attack_attrs, expert_indices, n_features)
            rma = compute_rma(attack_attrs, expert_indices)

            results.append(AlignmentResult(
                attack_type=label_name,
                explainer_key=exp_key,
                rra_score=rra,
                rma_score=rma,
                expert_features=expert_features,
                resolved_indices=expert_indices,
                n_expert_features=len(expert_features),
                n_resolved=len(expert_indices),
            ))

    return results


def alignment_to_dict(results: list[AlignmentResult]) -> list[dict]:
    """Convert alignment results to JSON-serializable dicts."""
    return [
        {
            "attack_type": r.attack_type,
            "explainer_key": r.explainer_key,
            "rra_score": r.rra_score,
            "rma_score": r.rma_score,
            "expert_features": r.expert_features,
            "n_expert_features": r.n_expert_features,
            "n_resolved": r.n_resolved,
        }
        for r in results
    ]
