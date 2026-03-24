"""Domain-aware feature constraints for adversarial perturbation bounding.

Defines per-dataset constraint specifications and a projection function that ensures
adversarial examples respect physical network traffic constraints (binary flags,
integer counts, one-hot encoding, ordering invariants, etc.).

Constraints operate in MinMax-scaled [0,1] space to integrate directly with FGSM/PGD.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────


class FeatureType(Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"
    RATE = "rate"
    ONE_HOT_MEMBER = "one_hot"


@dataclass
class FeatureBound:
    """Constraint for a single feature in ORIGINAL (pre-scaling) space."""

    name: str
    feature_type: FeatureType
    original_min: float = 0.0
    original_max: float = np.inf
    one_hot_group: str | None = None


@dataclass
class OrderingConstraint:
    """Enforces min_idx <= mean_idx <= max_idx in scaled space."""

    min_idx: int
    mean_idx: int
    max_idx: int


@dataclass
class FeatureConstraintSpec:
    """Complete constraint specification for one dataset."""

    dataset_name: str
    bounds: list[FeatureBound]
    one_hot_groups: dict[str, list[int]] = field(default_factory=dict)
    ordering_constraints: list[OrderingConstraint] = field(default_factory=list)


# ─── Per-Dataset Constraint Builders ─────────────────────────────────────────

# NSL-KDD feature sets
_NSL_KDD_BINARY = {
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
}

_NSL_KDD_INTEGER_COUNTS = {
    "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "src_bytes", "dst_bytes", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
}

_NSL_KDD_RATES = {
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
}

_NSL_KDD_ONE_HOT_PREFIXES = ["protocol_type_", "service_", "flag_"]


def build_nsl_kdd_constraints(
    feature_names: list[str], scaler: MinMaxScaler
) -> FeatureConstraintSpec:
    """Build constraints for NSL-KDD dataset (post one-hot encoding)."""
    bounds = []
    one_hot_groups: dict[str, list[int]] = {}

    for i, name in enumerate(feature_names):
        # One-hot encoded columns
        oh_match = None
        for prefix in _NSL_KDD_ONE_HOT_PREFIXES:
            if name.startswith(prefix):
                oh_match = prefix.rstrip("_")
                break

        if oh_match:
            bounds.append(FeatureBound(
                name=name,
                feature_type=FeatureType.ONE_HOT_MEMBER,
                original_min=0.0,
                original_max=1.0,
                one_hot_group=oh_match,
            ))
            one_hot_groups.setdefault(oh_match, []).append(i)

        elif name in _NSL_KDD_BINARY:
            bounds.append(FeatureBound(
                name=name,
                feature_type=FeatureType.BINARY,
                original_min=0.0,
                original_max=1.0,
            ))

        elif name in _NSL_KDD_RATES:
            bounds.append(FeatureBound(
                name=name,
                feature_type=FeatureType.RATE,
                original_min=0.0,
                original_max=1.0,
            ))

        elif name in _NSL_KDD_INTEGER_COUNTS:
            bounds.append(FeatureBound(
                name=name,
                feature_type=FeatureType.INTEGER,
                original_min=0.0,
                original_max=scaler.data_max_[i],
            ))

        elif name == "duration":
            bounds.append(FeatureBound(
                name=name,
                feature_type=FeatureType.CONTINUOUS,
                original_min=0.0,
                original_max=scaler.data_max_[i],
            ))

        else:
            # Fallback: treat as continuous with observed bounds
            bounds.append(FeatureBound(
                name=name,
                feature_type=FeatureType.CONTINUOUS,
                original_min=0.0,
                original_max=scaler.data_max_[i],
            ))

    spec = FeatureConstraintSpec(
        dataset_name="nsl-kdd",
        bounds=bounds,
        one_hot_groups=one_hot_groups,
    )

    logger.info(
        f"  NSL-KDD constraints: {len(bounds)} features, "
        f"{len(one_hot_groups)} one-hot groups, "
        f"{sum(1 for b in bounds if b.feature_type == FeatureType.BINARY)} binary, "
        f"{sum(1 for b in bounds if b.feature_type == FeatureType.INTEGER)} integer, "
        f"{sum(1 for b in bounds if b.feature_type == FeatureType.RATE)} rate"
    )
    return spec


# CICFlowMeter name-matching patterns
_CICFM_PORT_PATTERNS = [
    re.compile(r"(?i)^(destination\s*port|dst\s*port)$"),
]

_CICFM_PACKET_COUNT_PATTERNS = [
    re.compile(r"(?i)(total\s*(fwd|bwd|backward)\s*packet)"),
    re.compile(r"(?i)(tot\s*(fwd|bwd)\s*pkt)"),
    re.compile(r"(?i)(subflow\s*(fwd|bwd)\s*packet)"),
    re.compile(r"(?i)(subflow\s*(fwd|bwd)\s*pkt)"),
    re.compile(r"(?i)(act.data.pkt)"),
    re.compile(r"(?i)(fwd\s*act\s*data\s*pkt)"),
]

_CICFM_BYTE_COUNT_PATTERNS = [
    re.compile(r"(?i)(total\s*length\s*of\s*(fwd|bwd))"),
    re.compile(r"(?i)(totlen\s*(fwd|bwd))"),
    re.compile(r"(?i)(subflow\s*(fwd|bwd)\s*(byte|byt))"),
]

_CICFM_FLAG_PATTERNS = [
    re.compile(r"(?i)(fin|syn|rst|psh|ack|urg|cwe?|ece)\s*flag\s*(count|cnt)"),
    re.compile(r"(?i)(fwd|bwd)\s*(psh|urg)\s*flag"),
]

_CICFM_WINDOW_PATTERNS = [
    re.compile(r"(?i)(init.*(fwd|bwd)?\s*win)"),
]

_CICFM_HEADER_PATTERNS = [
    re.compile(r"(?i)(fwd|bwd)\s*header\s*(len|length)"),
]

_CICFM_SEG_MIN_PATTERNS = [
    re.compile(r"(?i)(min.seg.size|fwd.seg.size.min)"),
]

_CICFM_RATE_PATTERNS = [
    re.compile(r"(?i)(byte|byt)s?/s"),
    re.compile(r"(?i)(packet|pkt)s?/s"),
    re.compile(r"(?i)bulk\s*rate"),
    re.compile(r"(?i)(fwd|bwd)\s*(avg|average)\s*(byte|byt|packet|pkt)s?/bulk"),
]

_CICFM_RATIO_PATTERNS = [
    re.compile(r"(?i)down.?up\s*ratio"),
]

# Ordering constraint detection patterns: (min_pattern, mean_pattern, max_pattern)
_CICFM_ORDERING_GROUPS = [
    # Packet lengths
    (r"(?i)fwd\s*(pkt|packet)\s*(len|length)\s*min",
     r"(?i)fwd\s*(pkt|packet)\s*(len|length)\s*mean",
     r"(?i)fwd\s*(pkt|packet)\s*(len|length)\s*max"),
    (r"(?i)bwd\s*(pkt|packet)\s*(len|length)\s*min",
     r"(?i)bwd\s*(pkt|packet)\s*(len|length)\s*mean",
     r"(?i)bwd\s*(pkt|packet)\s*(len|length)\s*max"),
    (r"(?i)^(pkt|packet|min\s*packet)\s*(len|length)\s*min",
     r"(?i)^(pkt|packet)\s*(len|length)\s*mean",
     r"(?i)^(pkt|packet|max\s*packet)\s*(len|length)\s*max"),
    # Flow IAT
    (r"(?i)flow\s*iat\s*min", r"(?i)flow\s*iat\s*mean", r"(?i)flow\s*iat\s*max"),
    # Fwd IAT
    (r"(?i)fwd\s*iat\s*min", r"(?i)fwd\s*iat\s*mean", r"(?i)fwd\s*iat\s*max"),
    # Bwd IAT
    (r"(?i)bwd\s*iat\s*min", r"(?i)bwd\s*iat\s*mean", r"(?i)bwd\s*iat\s*max"),
    # Active times
    (r"(?i)active\s*min", r"(?i)active\s*mean", r"(?i)active\s*max"),
    # Idle times
    (r"(?i)idle\s*min", r"(?i)idle\s*mean", r"(?i)idle\s*max"),
]


def _match_any(name: str, patterns: list[re.Pattern]) -> bool:
    """Check if name matches any pattern in the list."""
    return any(p.search(name) for p in patterns)


def _find_index(feature_names: list[str], pattern_str: str) -> int | None:
    """Find the index of the first feature matching the regex pattern."""
    pat = re.compile(pattern_str)
    for i, name in enumerate(feature_names):
        if pat.search(name.strip()):
            return i
    return None


def build_cicflowmeter_constraints(
    feature_names: list[str], scaler: MinMaxScaler, dataset_name: str
) -> FeatureConstraintSpec:
    """Build constraints for CICFlowMeter-based datasets (CIC-IDS-2017, UNSW-NB15, CSE-CIC-IDS2018)."""
    bounds = []
    counts = {"port": 0, "pkt_count": 0, "byte_count": 0, "flag": 0,
              "window": 0, "header": 0, "rate": 0, "integer_other": 0, "continuous": 0}

    for i, raw_name in enumerate(feature_names):
        name = raw_name.strip()

        if _match_any(name, _CICFM_PORT_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=0.0, original_max=65535.0))
            counts["port"] += 1

        elif _match_any(name, _CICFM_PACKET_COUNT_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["pkt_count"] += 1

        elif _match_any(name, _CICFM_BYTE_COUNT_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["byte_count"] += 1

        elif _match_any(name, _CICFM_FLAG_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["flag"] += 1

        elif _match_any(name, _CICFM_WINDOW_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=-1.0, original_max=65535.0))
            counts["window"] += 1

        elif _match_any(name, _CICFM_HEADER_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["header"] += 1

        elif _match_any(name, _CICFM_SEG_MIN_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.INTEGER,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["integer_other"] += 1

        elif _match_any(name, _CICFM_RATE_PATTERNS) or _match_any(name, _CICFM_RATIO_PATTERNS):
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.CONTINUOUS,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["rate"] += 1

        else:
            # Remaining: IAT, Active/Idle, Duration, packet length stats, averages
            bounds.append(FeatureBound(name=name, feature_type=FeatureType.CONTINUOUS,
                                       original_min=0.0, original_max=scaler.data_max_[i]))
            counts["continuous"] += 1

    # Build ordering constraints
    ordering = []
    for min_pat, mean_pat, max_pat in _CICFM_ORDERING_GROUPS:
        min_idx = _find_index(feature_names, min_pat)
        mean_idx = _find_index(feature_names, mean_pat)
        max_idx = _find_index(feature_names, max_pat)
        if min_idx is not None and mean_idx is not None and max_idx is not None:
            ordering.append(OrderingConstraint(min_idx, mean_idx, max_idx))

    spec = FeatureConstraintSpec(
        dataset_name=dataset_name,
        bounds=bounds,
        ordering_constraints=ordering,
    )

    logger.info(
        f"  {dataset_name} constraints: {len(bounds)} features — "
        f"port={counts['port']}, pkt_count={counts['pkt_count']}, "
        f"byte_count={counts['byte_count']}, flag={counts['flag']}, "
        f"window={counts['window']}, header={counts['header']}, "
        f"rate={counts['rate']}, integer_other={counts['integer_other']}, "
        f"continuous={counts['continuous']}, "
        f"ordering_constraints={len(ordering)}"
    )
    return spec


def build_cic_iov_2024_constraints(
    feature_names: list[str], scaler: MinMaxScaler
) -> FeatureConstraintSpec:
    """Build constraints for CIC-IoV-2024 (obfuscated DATA_0..DATA_7, integer [0,255])."""
    bounds = []
    for name in feature_names:
        bounds.append(FeatureBound(
            name=name.strip(),
            feature_type=FeatureType.INTEGER,
            original_min=0.0,
            original_max=255.0,
        ))

    spec = FeatureConstraintSpec(dataset_name="cic-iov-2024", bounds=bounds)
    logger.info(f"  CIC-IoV-2024 constraints: {len(bounds)} features, all INTEGER [0, 255]")
    return spec


def build_constraints(
    dataset_name: str, feature_names: list[str], scaler: MinMaxScaler
) -> FeatureConstraintSpec:
    """Top-level factory: build constraint spec for any supported dataset."""
    logger.info(f"Building feature constraints for {dataset_name}...")

    if dataset_name == "nsl-kdd":
        return build_nsl_kdd_constraints(feature_names, scaler)
    elif dataset_name in ("cic-ids-2017", "unsw-nb15", "cse-cic-ids2018"):
        return build_cicflowmeter_constraints(feature_names, scaler, dataset_name)
    elif dataset_name == "cic-iov-2024":
        return build_cic_iov_2024_constraints(feature_names, scaler)
    else:
        raise ValueError(f"No constraints defined for dataset: {dataset_name}")


# ─── Scaled-Space Bound Computation ──────────────────────────────────────────


def compute_scaled_bounds(
    spec: FeatureConstraintSpec, scaler: MinMaxScaler
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature [lower, upper] bounds in MinMax-scaled [0,1] space.

    Maps original domain bounds through the scaler:
        scaled = (original - data_min) / (data_max - data_min)
    """
    n = len(spec.bounds)
    lower = np.zeros(n, dtype=np.float32)
    upper = np.ones(n, dtype=np.float32)

    scale = scaler.data_max_ - scaler.data_min_

    for i, fb in enumerate(spec.bounds):
        if scale[i] < 1e-12:
            # Constant feature: freeze at 0 (its only scaled value)
            lower[i] = 0.0
            upper[i] = 0.0
            continue

        lo = (fb.original_min - scaler.data_min_[i]) / scale[i]
        hi = (fb.original_max - scaler.data_min_[i]) / scale[i]
        lower[i] = max(0.0, lo)
        upper[i] = min(1.0, hi)

    return lower, upper


# ─── Projection Function ─────────────────────────────────────────────────────


def project_constraints(
    X_adv: torch.Tensor,
    X_original: torch.Tensor,
    epsilon: float,
    spec: FeatureConstraintSpec,
    scaled_lower: torch.Tensor,
    scaled_upper: torch.Tensor,
    binary_indices: torch.Tensor,
    binary_scaled_0: torch.Tensor,
    binary_scaled_1: torch.Tensor,
    integer_indices: torch.Tensor,
    data_min: torch.Tensor,
    scale: torch.Tensor,
    int_original_min: torch.Tensor,
    int_original_max: torch.Tensor,
    one_hot_groups: dict[str, torch.Tensor],
    oh_scaled_0: float,
    oh_scaled_1: float,
    constant_mask: torch.Tensor,
) -> torch.Tensor:
    """Project adversarial examples onto the feasible constraint set.

    Operates in MinMax-scaled [0,1] space. Projection order:
    1. Per-feature domain bounds
    2. Binary feature snapping
    3. Integer feature snapping (inverse-scale, round, re-scale)
    4. One-hot group projection (argmax re-encoding)
    5. Ordering constraints (sort triplets)
    6. Freeze constant features
    7. L-inf epsilon-ball re-projection
    """
    # 1. Per-feature bound clamping
    X_adv = torch.max(X_adv, scaled_lower.unsqueeze(0))
    X_adv = torch.min(X_adv, scaled_upper.unsqueeze(0))

    # 2. Binary feature snapping
    if binary_indices.numel() > 0:
        cols = X_adv[:, binary_indices]
        mid = (binary_scaled_0 + binary_scaled_1) / 2.0
        snapped = torch.where(cols >= mid.unsqueeze(0), binary_scaled_1.unsqueeze(0),
                              binary_scaled_0.unsqueeze(0))
        X_adv = X_adv.clone()
        X_adv[:, binary_indices] = snapped

    # 3. Integer feature snapping
    if integer_indices.numel() > 0:
        cols = X_adv[:, integer_indices]
        s = scale[integer_indices].unsqueeze(0)
        dm = data_min[integer_indices].unsqueeze(0)
        # Inverse scale to original space
        original_val = cols * s + dm
        # Round and clamp to original bounds
        rounded = torch.round(original_val)
        rounded = torch.max(rounded, int_original_min.unsqueeze(0))
        rounded = torch.min(rounded, int_original_max.unsqueeze(0))
        # Re-scale
        X_adv = X_adv.clone()
        X_adv[:, integer_indices] = (rounded - dm) / s

    # 4. One-hot group projection
    for _group_name, col_indices in one_hot_groups.items():
        group_cols = X_adv[:, col_indices]
        winner = group_cols.argmax(dim=1)
        new_vals = torch.full_like(group_cols, oh_scaled_0)
        new_vals.scatter_(1, winner.unsqueeze(1), oh_scaled_1)
        X_adv = X_adv.clone()
        X_adv[:, col_indices] = new_vals

    # 5. Ordering constraints (min <= mean <= max)
    for oc in spec.ordering_constraints:
        triplet = X_adv[:, [oc.min_idx, oc.mean_idx, oc.max_idx]]
        sorted_vals, _ = torch.sort(triplet, dim=1)
        X_adv = X_adv.clone()
        X_adv[:, oc.min_idx] = sorted_vals[:, 0]
        X_adv[:, oc.mean_idx] = sorted_vals[:, 1]
        X_adv[:, oc.max_idx] = sorted_vals[:, 2]

    # 6. Freeze constant features (zero scale)
    if constant_mask.any():
        X_adv = X_adv.clone()
        X_adv[:, constant_mask] = X_original[:, constant_mask]

    # 7. L-inf epsilon-ball re-projection
    delta = (X_adv - X_original).clamp(-epsilon, epsilon)
    X_adv = (X_original + delta).clamp(0.0, 1.0)

    return X_adv


# ─── Factory: Build Projector Closure ─────────────────────────────────────────


def make_constraint_projector(
    spec: FeatureConstraintSpec,
    scaler: MinMaxScaler,
    device: torch.device,
):
    """Create a constraint projection closure for use in FGSM/PGD attacks.

    Returns:
        callable(X_adv: Tensor, X_original: Tensor, epsilon: float) -> Tensor
    """
    lower_np, upper_np = compute_scaled_bounds(spec, scaler)
    scaled_lower = torch.tensor(lower_np, dtype=torch.float32, device=device)
    scaled_upper = torch.tensor(upper_np, dtype=torch.float32, device=device)

    scale_np = (scaler.data_max_ - scaler.data_min_).astype(np.float32)
    data_min_np = scaler.data_min_.astype(np.float32)
    data_min_t = torch.tensor(data_min_np, dtype=torch.float32, device=device)
    scale_t = torch.tensor(scale_np, dtype=torch.float32, device=device)
    # Prevent division by zero for constant features
    scale_safe = scale_t.clone()
    scale_safe[scale_safe < 1e-12] = 1.0

    # Constant feature mask
    constant_mask = torch.tensor(scale_np < 1e-12, dtype=torch.bool, device=device)

    # Binary feature indices and scaled values
    binary_idx = []
    binary_s0 = []
    binary_s1 = []
    for i, fb in enumerate(spec.bounds):
        if fb.feature_type == FeatureType.BINARY:
            if scale_np[i] < 1e-12:
                continue  # constant, handled by freeze
            binary_idx.append(i)
            binary_s0.append((0.0 - data_min_np[i]) / scale_np[i])
            binary_s1.append((1.0 - data_min_np[i]) / scale_np[i])

    binary_indices = torch.tensor(binary_idx, dtype=torch.long, device=device)
    binary_scaled_0 = torch.tensor(binary_s0, dtype=torch.float32, device=device) if binary_s0 else torch.empty(0, device=device)
    binary_scaled_1 = torch.tensor(binary_s1, dtype=torch.float32, device=device) if binary_s1 else torch.empty(0, device=device)

    # Integer feature indices and original bounds
    int_idx = []
    int_omin = []
    int_omax = []
    for i, fb in enumerate(spec.bounds):
        if fb.feature_type == FeatureType.INTEGER:
            if scale_np[i] < 1e-12:
                continue
            int_idx.append(i)
            int_omin.append(fb.original_min)
            int_omax.append(fb.original_max if np.isfinite(fb.original_max) else scaler.data_max_[i])

    integer_indices = torch.tensor(int_idx, dtype=torch.long, device=device)
    int_original_min = torch.tensor(int_omin, dtype=torch.float32, device=device) if int_omin else torch.empty(0, device=device)
    int_original_max = torch.tensor(int_omax, dtype=torch.float32, device=device) if int_omax else torch.empty(0, device=device)

    # One-hot groups as tensors
    oh_groups_t: dict[str, torch.Tensor] = {}
    oh_scaled_0 = 0.0
    oh_scaled_1 = 1.0
    for group_name, indices in spec.one_hot_groups.items():
        oh_groups_t[group_name] = torch.tensor(indices, dtype=torch.long, device=device)
        # Compute scaled values for one-hot (use first non-constant column)
        for idx in indices:
            if scale_np[idx] >= 1e-12:
                oh_scaled_0 = float((0.0 - data_min_np[idx]) / scale_np[idx])
                oh_scaled_1 = float((1.0 - data_min_np[idx]) / scale_np[idx])
                break

    def projector(
        X_adv: torch.Tensor,
        X_original: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        return project_constraints(
            X_adv=X_adv,
            X_original=X_original,
            epsilon=epsilon,
            spec=spec,
            scaled_lower=scaled_lower,
            scaled_upper=scaled_upper,
            binary_indices=binary_indices,
            binary_scaled_0=binary_scaled_0,
            binary_scaled_1=binary_scaled_1,
            integer_indices=integer_indices,
            data_min=data_min_t,
            scale=scale_safe,
            int_original_min=int_original_min,
            int_original_max=int_original_max,
            one_hot_groups=oh_groups_t,
            oh_scaled_0=oh_scaled_0,
            oh_scaled_1=oh_scaled_1,
            constant_mask=constant_mask,
        )

    return projector


def spec_to_dict(spec: FeatureConstraintSpec) -> dict:
    """Serialize a FeatureConstraintSpec to a JSON-compatible dict."""
    return {
        "dataset_name": spec.dataset_name,
        "num_features": len(spec.bounds),
        "feature_types": {
            ft.value: sum(1 for b in spec.bounds if b.feature_type == ft)
            for ft in FeatureType
        },
        "one_hot_groups": {k: v for k, v in spec.one_hot_groups.items()},
        "num_ordering_constraints": len(spec.ordering_constraints),
        "ordering_constraints": [
            {"min_idx": oc.min_idx, "mean_idx": oc.mean_idx, "max_idx": oc.max_idx}
            for oc in spec.ordering_constraints
        ],
        "bounds_summary": [
            {"name": b.name, "type": b.feature_type.value,
             "original_min": float(b.original_min),
             "original_max": float(b.original_max) if np.isfinite(b.original_max) else "inf"}
            for b in spec.bounds
        ],
    }
