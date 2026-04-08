"""Dataset schemas for PA-XAI: protocol-aware feature metadata for NIDS datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Protocol integer encodings (as used by CICFlowMeter)
TCP_PROTOCOL_INT: int = 6
UDP_PROTOCOL_INT: int = 17
ICMP_PROTOCOL_INT: int = 1

# Sentinel set for string-encoded protocol detection:
# NSL-KDD encodes protocol_type as 0=tcp, 1=udp, 2=icmp after label encoding
_STRING_ENCODED_PROTOCOL_SENTINEL: set[int] = {0, 1, 2}


@dataclass(frozen=True)
class HierarchicalConstraint:
    """Represents the constraint max_feature >= mean_feature >= min_feature.

    Attributes:
        max_feature: Name of the maximum-value feature.
        mean_feature: Name of the mean-value feature.
        min_feature: Name of the minimum-value feature.
    """

    max_feature: str
    mean_feature: str
    min_feature: str


@dataclass(frozen=True)
class BoundedRangeConstraint:
    """Clamps a feature to [lower, upper]."""
    feature: str
    lower: float
    upper: float


@dataclass(frozen=True)
class CrossFeatureConstraint:
    """Enforces arithmetic relationships between features.

    Relations:
      - "ratio": derived = operands[0] / operands[1]
      - "sum_ratio": derived = (operands[0] + operands[1]) / operands[2]
      - "square": derived = operands[0] ** 2
      - "equal": derived = operands[0]
    """
    derived_feature: str
    relation: str
    operands: list[str]


@dataclass(frozen=True)
class StdRangeConstraint:
    """Enforces std <= max - min for a statistical group."""
    std_feature: str
    max_feature: str
    min_feature: str


@dataclass
class DatasetSchema:
    """Metadata schema for a network intrusion detection dataset.

    Attributes:
        name: Human-readable dataset identifier.
        feature_names: Ordered list of feature column names.
        protocol_feature: Name of the protocol column, or None if absent.
        non_negative_features: Features that are physically non-negative.
        tcp_only_features: Features that are only meaningful for TCP flows.
        discrete_features: Features with discrete/categorical values.
        hierarchical_constraints: List of max >= mean >= min constraints.
        protocol_encoding: One of "integer", "string", or "auto".

    Computed attributes (set in __post_init__):
        protocol_index: Index of protocol_feature in feature_names, or None.
        non_negative_indices: Indices of non_negative_features.
        tcp_only_indices: Indices of tcp_only_features.
        discrete_indices: Indices of discrete_features.
        hierarchical_index_triples: List of (max_idx, mean_idx, min_idx) tuples.
    """

    # Init fields
    name: str
    feature_names: list[str]
    protocol_feature: Optional[str]
    non_negative_features: list[str]
    tcp_only_features: list[str]
    discrete_features: list[str]
    hierarchical_constraints: list[HierarchicalConstraint]
    protocol_encoding: str
    bounded_range_constraints: list[BoundedRangeConstraint] = field(default_factory=list)
    cross_feature_constraints: list[CrossFeatureConstraint] = field(default_factory=list)
    std_range_constraints: list[StdRangeConstraint] = field(default_factory=list)
    udp_only_features: list[str] = field(default_factory=list)
    icmp_only_features: list[str] = field(default_factory=list)
    connection_only_features: list[str] = field(default_factory=list)
    duplicate_features: list[tuple[str, str]] = field(default_factory=list)

    # Computed fields
    protocol_index: Optional[int] = field(init=False)
    non_negative_indices: list[int] = field(init=False)
    tcp_only_indices: list[int] = field(init=False)
    discrete_indices: list[int] = field(init=False)
    hierarchical_index_triples: list[tuple[int, int, int]] = field(init=False)
    bounded_range_index_bounds: list[tuple[int, float, float]] = field(init=False)
    cross_feature_index_tuples: list[tuple] = field(init=False)
    std_range_index_triples: list[tuple[int, int, int]] = field(init=False)
    udp_only_indices: list[int] = field(init=False)
    icmp_only_indices: list[int] = field(init=False)
    connection_only_indices: list[int] = field(init=False)
    duplicate_index_pairs: list[tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        valid_encodings = {"integer", "string", "auto"}
        if self.protocol_encoding not in valid_encodings:
            raise ValueError(
                f"protocol_encoding must be one of {valid_encodings!r}, "
                f"got {self.protocol_encoding!r}"
            )

        fn = self.feature_names
        idx = {name: i for i, name in enumerate(fn)}

        self.protocol_index = idx.get(self.protocol_feature) if self.protocol_feature else None
        self.non_negative_indices = [idx[f] for f in self.non_negative_features if f in idx]
        self.tcp_only_indices = [idx[f] for f in self.tcp_only_features if f in idx]
        self.discrete_indices = [idx[f] for f in self.discrete_features if f in idx]
        self.hierarchical_index_triples = [
            (idx[hc.max_feature], idx[hc.mean_feature], idx[hc.min_feature])
            for hc in self.hierarchical_constraints
            if hc.max_feature in idx and hc.mean_feature in idx and hc.min_feature in idx
        ]
        self.bounded_range_index_bounds = [
            (idx[brc.feature], brc.lower, brc.upper)
            for brc in self.bounded_range_constraints
            if brc.feature in idx
        ]
        self.cross_feature_index_tuples = []
        for cfc in self.cross_feature_constraints:
            if cfc.derived_feature not in idx:
                continue
            operand_indices = [idx[op] for op in cfc.operands if op in idx]
            if len(operand_indices) == len(cfc.operands):
                self.cross_feature_index_tuples.append(
                    (idx[cfc.derived_feature], operand_indices, cfc.relation)
                )
        self.std_range_index_triples = [
            (idx[src.std_feature], idx[src.max_feature], idx[src.min_feature])
            for src in self.std_range_constraints
            if src.std_feature in idx and src.max_feature in idx and src.min_feature in idx
        ]
        self.udp_only_indices = [idx[f] for f in self.udp_only_features if f in idx]
        self.icmp_only_indices = [idx[f] for f in self.icmp_only_features if f in idx]
        self.connection_only_indices = [idx[f] for f in self.connection_only_features if f in idx]
        self.duplicate_index_pairs = [
            (idx[a], idx[b])
            for a, b in self.duplicate_features
            if a in idx and b in idx
        ]


def detect_protocol_encoding(
    row: np.ndarray,
    protocol_feature: str,
    feature_names: list[str],
) -> str:
    """Heuristically detect whether the protocol column uses integer or string encoding.

    CICFlowMeter datasets use integer protocol numbers (TCP=6, UDP=17, ICMP=1).
    NSL-KDD and UNSW-NB15 native use string-encoded protocols (tcp, udp, icmp)
    which after label-encoding map to small integers {0, 1, 2}.

    A protocol value in {0, 1, 2} is treated as string-encoded; otherwise integer.

    Args:
        row: A single sample as a 1-D numpy array.
        protocol_feature: The name of the protocol column.
        feature_names: Ordered list of feature names matching row indices.

    Returns:
        "integer" or "string"
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    proto_idx = idx.get(protocol_feature)
    if proto_idx is None:
        return "integer"

    proto_val = float(row[proto_idx])
    if proto_val in _STRING_ENCODED_PROTOCOL_SENTINEL:
        return "string"
    return "integer"


def _is_tcp(protocol_value: float, encoding: str, tcp_int: int = TCP_PROTOCOL_INT) -> bool:
    """Return True if the given protocol value represents TCP.

    Args:
        protocol_value: Numeric value of the protocol feature.
        encoding: "integer" or "string" encoding mode.
        tcp_int: The integer value for TCP (default 6).
    """
    if encoding == "integer":
        return int(protocol_value) == tcp_int
    # string encoding: TCP maps to 0 after label encoding (alphabetical: tcp < udp)
    return int(protocol_value) == 0


# ---------------------------------------------------------------------------
# CIC-IDS-2017
# ---------------------------------------------------------------------------

_CIC_2017_FEATURES = [
    "Destination Port", "Protocol", "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_CIC_2017_TCP_ONLY = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]

_CIC_2017_NON_NEGATIVE = [
    "Flow Duration",
    "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size", "Avg Bwd Segment Size",
    "Fwd Header Length.1",
    "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_CIC_2017_DISCRETE = [
    "Destination Port", "Protocol",
    "Total Fwd Packets", "Total Backward Packets",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWE Flag Count", "ECE Flag Count",
    "Subflow Fwd Packets", "Subflow Bwd Packets",
    "act_data_pkt_fwd",
]

_CIC_2017_HIERARCHICAL = [
    HierarchicalConstraint("Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Min"),
    HierarchicalConstraint("Bwd Packet Length Max", "Bwd Packet Length Mean", "Bwd Packet Length Min"),
    HierarchicalConstraint("Max Packet Length", "Packet Length Mean", "Min Packet Length"),
    HierarchicalConstraint("Flow IAT Max", "Flow IAT Mean", "Flow IAT Min"),
    HierarchicalConstraint("Fwd IAT Max", "Fwd IAT Mean", "Fwd IAT Min"),
    HierarchicalConstraint("Bwd IAT Max", "Bwd IAT Mean", "Bwd IAT Min"),
    HierarchicalConstraint("Active Max", "Active Mean", "Active Min"),
    HierarchicalConstraint("Idle Max", "Idle Mean", "Idle Min"),
]

_CIC_2017_BOUNDED_RANGE = [
    BoundedRangeConstraint("Destination Port", 0.0, 65535.0),
    BoundedRangeConstraint("Init_Win_bytes_forward", 0.0, 65535.0),
    BoundedRangeConstraint("Init_Win_bytes_backward", 0.0, 65535.0),
]

_CIC_2017_CROSS_FEATURE = [
    CrossFeatureConstraint("Flow Bytes/s", "sum_ratio", ["Total Length of Fwd Packets", "Total Length of Bwd Packets", "Flow Duration"]),
    CrossFeatureConstraint("Flow Packets/s", "sum_ratio", ["Total Fwd Packets", "Total Backward Packets", "Flow Duration"]),
    CrossFeatureConstraint("Fwd Packets/s", "ratio", ["Total Fwd Packets", "Flow Duration"]),
    CrossFeatureConstraint("Bwd Packets/s", "ratio", ["Total Backward Packets", "Flow Duration"]),
    CrossFeatureConstraint("Packet Length Variance", "square", ["Packet Length Std"]),
    CrossFeatureConstraint("Subflow Fwd Packets", "equal", ["Total Fwd Packets"]),
    CrossFeatureConstraint("Subflow Fwd Bytes", "equal", ["Total Length of Fwd Packets"]),
    CrossFeatureConstraint("Subflow Bwd Packets", "equal", ["Total Backward Packets"]),
    CrossFeatureConstraint("Subflow Bwd Bytes", "equal", ["Total Length of Bwd Packets"]),
]

_CIC_2017_STD_RANGE = [
    StdRangeConstraint("Fwd Packet Length Std", "Fwd Packet Length Max", "Fwd Packet Length Min"),
    StdRangeConstraint("Bwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min"),
    StdRangeConstraint("Packet Length Std", "Max Packet Length", "Min Packet Length"),
    StdRangeConstraint("Flow IAT Std", "Flow IAT Max", "Flow IAT Min"),
    StdRangeConstraint("Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min"),
    StdRangeConstraint("Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"),
    StdRangeConstraint("Active Std", "Active Max", "Active Min"),
    StdRangeConstraint("Idle Std", "Idle Max", "Idle Min"),
]

CIC_IDS_2017 = DatasetSchema(
    name="CIC-IDS-2017",
    feature_names=_CIC_2017_FEATURES,
    protocol_feature="Protocol",
    non_negative_features=_CIC_2017_NON_NEGATIVE,
    tcp_only_features=_CIC_2017_TCP_ONLY,
    discrete_features=_CIC_2017_DISCRETE,
    hierarchical_constraints=_CIC_2017_HIERARCHICAL,
    protocol_encoding="integer",
    bounded_range_constraints=_CIC_2017_BOUNDED_RANGE,
    cross_feature_constraints=_CIC_2017_CROSS_FEATURE,
    std_range_constraints=_CIC_2017_STD_RANGE,
    duplicate_features=[("Fwd Header Length", "Fwd Header Length.1")],
)


# ---------------------------------------------------------------------------
# CSE-CIC-IDS2018
# ---------------------------------------------------------------------------

_CIC_2018_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_CIC_2018_TCP_ONLY = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt",
    "Init Fwd Win Byts", "Init Bwd Win Byts",
]

_CIC_2018_NON_NEGATIVE = [
    "Flow Duration",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts",
    "Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
    "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
    "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len",
    "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg",
    "Bwd Byts/b Avg", "Bwd Pkts/b Avg", "Bwd Blk Rate Avg",
    "Subflow Fwd Pkts", "Subflow Fwd Byts", "Subflow Bwd Pkts", "Subflow Bwd Byts",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_CIC_2018_DISCRETE = [
    "Dst Port", "Protocol",
    "Tot Fwd Pkts", "Tot Bwd Pkts",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt", "ACK Flag Cnt", "URG Flag Cnt",
    "CWE Flag Count", "ECE Flag Cnt",
    "Subflow Fwd Pkts", "Subflow Bwd Pkts",
    "Fwd Act Data Pkts",
]

_CIC_2018_HIERARCHICAL = [
    HierarchicalConstraint("Fwd Pkt Len Max", "Fwd Pkt Len Mean", "Fwd Pkt Len Min"),
    HierarchicalConstraint("Bwd Pkt Len Max", "Bwd Pkt Len Mean", "Bwd Pkt Len Min"),
    HierarchicalConstraint("Pkt Len Max", "Pkt Len Mean", "Pkt Len Min"),
    HierarchicalConstraint("Flow IAT Max", "Flow IAT Mean", "Flow IAT Min"),
    HierarchicalConstraint("Fwd IAT Max", "Fwd IAT Mean", "Fwd IAT Min"),
    HierarchicalConstraint("Bwd IAT Max", "Bwd IAT Mean", "Bwd IAT Min"),
    HierarchicalConstraint("Active Max", "Active Mean", "Active Min"),
    HierarchicalConstraint("Idle Max", "Idle Mean", "Idle Min"),
]

_CIC_2018_BOUNDED_RANGE = [
    BoundedRangeConstraint("Dst Port", 0.0, 65535.0),
    BoundedRangeConstraint("Init Fwd Win Byts", 0.0, 65535.0),
    BoundedRangeConstraint("Init Bwd Win Byts", 0.0, 65535.0),
]

_CIC_2018_CROSS_FEATURE = [
    CrossFeatureConstraint("Flow Byts/s", "sum_ratio", ["TotLen Fwd Pkts", "TotLen Bwd Pkts", "Flow Duration"]),
    CrossFeatureConstraint("Flow Pkts/s", "sum_ratio", ["Tot Fwd Pkts", "Tot Bwd Pkts", "Flow Duration"]),
    CrossFeatureConstraint("Fwd Pkts/s", "ratio", ["Tot Fwd Pkts", "Flow Duration"]),
    CrossFeatureConstraint("Bwd Pkts/s", "ratio", ["Tot Bwd Pkts", "Flow Duration"]),
    CrossFeatureConstraint("Pkt Len Var", "square", ["Pkt Len Std"]),
    CrossFeatureConstraint("Subflow Fwd Pkts", "equal", ["Tot Fwd Pkts"]),
    CrossFeatureConstraint("Subflow Fwd Byts", "equal", ["TotLen Fwd Pkts"]),
    CrossFeatureConstraint("Subflow Bwd Pkts", "equal", ["Tot Bwd Pkts"]),
    CrossFeatureConstraint("Subflow Bwd Byts", "equal", ["TotLen Bwd Pkts"]),
]

_CIC_2018_STD_RANGE = [
    StdRangeConstraint("Fwd Pkt Len Std", "Fwd Pkt Len Max", "Fwd Pkt Len Min"),
    StdRangeConstraint("Bwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min"),
    StdRangeConstraint("Pkt Len Std", "Pkt Len Max", "Pkt Len Min"),
    StdRangeConstraint("Flow IAT Std", "Flow IAT Max", "Flow IAT Min"),
    StdRangeConstraint("Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min"),
    StdRangeConstraint("Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"),
    StdRangeConstraint("Active Std", "Active Max", "Active Min"),
    StdRangeConstraint("Idle Std", "Idle Max", "Idle Min"),
]

CSE_CIC_IDS2018 = DatasetSchema(
    name="CSE-CIC-IDS2018",
    feature_names=_CIC_2018_FEATURES,
    protocol_feature="Protocol",
    non_negative_features=_CIC_2018_NON_NEGATIVE,
    tcp_only_features=_CIC_2018_TCP_ONLY,
    discrete_features=_CIC_2018_DISCRETE,
    hierarchical_constraints=_CIC_2018_HIERARCHICAL,
    protocol_encoding="integer",
    bounded_range_constraints=_CIC_2018_BOUNDED_RANGE,
    cross_feature_constraints=_CIC_2018_CROSS_FEATURE,
    std_range_constraints=_CIC_2018_STD_RANGE,
)


# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------

_NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

_NSL_KDD_NON_NEGATIVE = [
    "duration",
    "src_bytes", "dst_bytes", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "num_compromised",
    "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]

_NSL_KDD_DISCRETE = [
    "protocol_type", "service", "flag",
    "land", "wrong_fragment", "urgent",
    "num_failed_logins", "logged_in",
    "root_shell", "su_attempted",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]

_NSL_KDD_BOUNDED_RANGE = [
    BoundedRangeConstraint("serror_rate", 0.0, 1.0),
    BoundedRangeConstraint("srv_serror_rate", 0.0, 1.0),
    BoundedRangeConstraint("rerror_rate", 0.0, 1.0),
    BoundedRangeConstraint("srv_rerror_rate", 0.0, 1.0),
    BoundedRangeConstraint("same_srv_rate", 0.0, 1.0),
    BoundedRangeConstraint("diff_srv_rate", 0.0, 1.0),
    BoundedRangeConstraint("srv_diff_host_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_same_srv_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_diff_srv_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_same_src_port_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_srv_diff_host_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_serror_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_srv_serror_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_rerror_rate", 0.0, 1.0),
    BoundedRangeConstraint("dst_host_srv_rerror_rate", 0.0, 1.0),
    BoundedRangeConstraint("land", 0.0, 1.0),
    BoundedRangeConstraint("logged_in", 0.0, 1.0),
    BoundedRangeConstraint("root_shell", 0.0, 1.0),
    BoundedRangeConstraint("su_attempted", 0.0, 1.0),
    BoundedRangeConstraint("is_host_login", 0.0, 1.0),
    BoundedRangeConstraint("is_guest_login", 0.0, 1.0),
]

_NSL_KDD_CONNECTION_ONLY = [
    "num_failed_logins", "logged_in", "root_shell", "su_attempted",
    "num_shells", "num_access_files",
]

NSL_KDD = DatasetSchema(
    name="NSL-KDD",
    feature_names=_NSL_KDD_FEATURES,
    protocol_feature="protocol_type",
    non_negative_features=_NSL_KDD_NON_NEGATIVE,
    tcp_only_features=["urgent"],
    discrete_features=_NSL_KDD_DISCRETE,
    hierarchical_constraints=[],
    protocol_encoding="string",
    bounded_range_constraints=_NSL_KDD_BOUNDED_RANGE,
    connection_only_features=_NSL_KDD_CONNECTION_ONLY,
)


# ---------------------------------------------------------------------------
# UNSW-NB15 Native (Argus/Bro)
# ---------------------------------------------------------------------------

_UNSW_NATIVE_FEATURES = [
    "dur", "proto", "service", "state",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "Srcport", "Dstport",
]

_UNSW_NATIVE_TCP_ONLY = [
    "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
]

_UNSW_NATIVE_NON_NEGATIVE = [
    "dur",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin",
    "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "Srcport", "Dstport",
]

_UNSW_NATIVE_DISCRETE = [
    "proto", "service", "state",
    "spkts", "dpkts",
    "sloss", "dloss",
    "swin", "dwin",
    "trans_depth",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports", "Srcport", "Dstport",
]

_UNSW_NATIVE_BOUNDED_RANGE = [
    BoundedRangeConstraint("sttl", 0.0, 255.0),
    BoundedRangeConstraint("dttl", 0.0, 255.0),
    BoundedRangeConstraint("swin", 0.0, 65535.0),
    BoundedRangeConstraint("dwin", 0.0, 65535.0),
    BoundedRangeConstraint("Srcport", 0.0, 65535.0),
    BoundedRangeConstraint("Dstport", 0.0, 65535.0),
    BoundedRangeConstraint("is_ftp_login", 0.0, 1.0),
    BoundedRangeConstraint("is_sm_ips_ports", 0.0, 1.0),
]

_UNSW_NATIVE_CONNECTION_ONLY = [
    "trans_depth", "response_body_len", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd",
]

UNSW_NB15_NATIVE = DatasetSchema(
    name="UNSW-NB15-Native",
    feature_names=_UNSW_NATIVE_FEATURES,
    protocol_feature="proto",
    non_negative_features=_UNSW_NATIVE_NON_NEGATIVE,
    tcp_only_features=_UNSW_NATIVE_TCP_ONLY,
    discrete_features=_UNSW_NATIVE_DISCRETE,
    hierarchical_constraints=[],
    protocol_encoding="string",
    bounded_range_constraints=_UNSW_NATIVE_BOUNDED_RANGE,
    connection_only_features=_UNSW_NATIVE_CONNECTION_ONLY,
)


# ---------------------------------------------------------------------------
# UNSW-NB15 CICFlowMeter
# ---------------------------------------------------------------------------

_UNSW_CIC_FEATURES = [
    "Protocol", "Flow Duration",
    "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "FWD Init Win Bytes", "Bwd Init Win Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_UNSW_CIC_TCP_ONLY = [
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count",
    "FWD Init Win Bytes", "Bwd Init Win Bytes",
]

_UNSW_CIC_NON_NEGATIVE = [
    "Flow Duration",
    "Total Fwd Packet", "Total Bwd packets",
    "Total Length of Fwd Packet", "Total Length of Bwd Packet",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
    "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Length", "Bwd Header Length",
    "Fwd Packets/s", "Bwd Packets/s",
    "Packet Length Min", "Packet Length Max", "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count",
    "Down/Up Ratio", "Average Packet Size", "Fwd Segment Size Avg", "Bwd Segment Size Avg",
    "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
    "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
    "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]

_UNSW_CIC_DISCRETE = [
    "Protocol",
    "Total Fwd Packet", "Total Bwd packets",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
    "CWR Flag Count", "ECE Flag Count",
    "Subflow Fwd Packets", "Subflow Bwd Packets",
    "Fwd Act Data Pkts",
]

_UNSW_CIC_HIERARCHICAL = [
    HierarchicalConstraint("Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Min"),
    HierarchicalConstraint("Bwd Packet Length Max", "Bwd Packet Length Mean", "Bwd Packet Length Min"),
    HierarchicalConstraint("Packet Length Max", "Packet Length Mean", "Packet Length Min"),
    HierarchicalConstraint("Flow IAT Max", "Flow IAT Mean", "Flow IAT Min"),
    HierarchicalConstraint("Fwd IAT Max", "Fwd IAT Mean", "Fwd IAT Min"),
    HierarchicalConstraint("Bwd IAT Max", "Bwd IAT Mean", "Bwd IAT Min"),
    HierarchicalConstraint("Active Max", "Active Mean", "Active Min"),
    HierarchicalConstraint("Idle Max", "Idle Mean", "Idle Min"),
]

_UNSW_CIC_BOUNDED_RANGE = [
    BoundedRangeConstraint("FWD Init Win Bytes", 0.0, 65535.0),
    BoundedRangeConstraint("Bwd Init Win Bytes", 0.0, 65535.0),
]

_UNSW_CIC_CROSS_FEATURE = [
    CrossFeatureConstraint("Flow Bytes/s", "sum_ratio", ["Total Length of Fwd Packet", "Total Length of Bwd Packet", "Flow Duration"]),
    CrossFeatureConstraint("Flow Packets/s", "sum_ratio", ["Total Fwd Packet", "Total Bwd packets", "Flow Duration"]),
    CrossFeatureConstraint("Fwd Packets/s", "ratio", ["Total Fwd Packet", "Flow Duration"]),
    CrossFeatureConstraint("Bwd Packets/s", "ratio", ["Total Bwd packets", "Flow Duration"]),
    CrossFeatureConstraint("Packet Length Variance", "square", ["Packet Length Std"]),
    CrossFeatureConstraint("Subflow Fwd Packets", "equal", ["Total Fwd Packet"]),
    CrossFeatureConstraint("Subflow Fwd Bytes", "equal", ["Total Length of Fwd Packet"]),
    CrossFeatureConstraint("Subflow Bwd Packets", "equal", ["Total Bwd packets"]),
    CrossFeatureConstraint("Subflow Bwd Bytes", "equal", ["Total Length of Bwd Packet"]),
]

_UNSW_CIC_STD_RANGE = [
    StdRangeConstraint("Fwd Packet Length Std", "Fwd Packet Length Max", "Fwd Packet Length Min"),
    StdRangeConstraint("Bwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min"),
    StdRangeConstraint("Packet Length Std", "Packet Length Max", "Packet Length Min"),
    StdRangeConstraint("Flow IAT Std", "Flow IAT Max", "Flow IAT Min"),
    StdRangeConstraint("Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min"),
    StdRangeConstraint("Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min"),
    StdRangeConstraint("Active Std", "Active Max", "Active Min"),
    StdRangeConstraint("Idle Std", "Idle Max", "Idle Min"),
]

UNSW_NB15_CIC = DatasetSchema(
    name="UNSW-NB15-CICFlowMeter",
    feature_names=_UNSW_CIC_FEATURES,
    protocol_feature="Protocol",
    non_negative_features=_UNSW_CIC_NON_NEGATIVE,
    tcp_only_features=_UNSW_CIC_TCP_ONLY,
    discrete_features=_UNSW_CIC_DISCRETE,
    hierarchical_constraints=_UNSW_CIC_HIERARCHICAL,
    protocol_encoding="integer",
    bounded_range_constraints=_UNSW_CIC_BOUNDED_RANGE,
    cross_feature_constraints=_UNSW_CIC_CROSS_FEATURE,
    std_range_constraints=_UNSW_CIC_STD_RANGE,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_SCHEMAS: dict[str, DatasetSchema] = {
    CIC_IDS_2017.name: CIC_IDS_2017,
    CSE_CIC_IDS2018.name: CSE_CIC_IDS2018,
    NSL_KDD.name: NSL_KDD,
    UNSW_NB15_NATIVE.name: UNSW_NB15_NATIVE,
    UNSW_NB15_CIC.name: UNSW_NB15_CIC,
}


def get_schema(name: str) -> DatasetSchema:
    """Look up a built-in schema by name.

    Args:
        name: The schema name (e.g. "CIC-IDS-2017").

    Returns:
        The corresponding DatasetSchema instance.

    Raises:
        KeyError: If the name is not a known built-in schema.
    """
    if name not in BUILTIN_SCHEMAS:
        raise KeyError(
            f"Unknown schema {name!r}. Available schemas: {list(BUILTIN_SCHEMAS.keys())}"
        )
    return BUILTIN_SCHEMAS[name]
