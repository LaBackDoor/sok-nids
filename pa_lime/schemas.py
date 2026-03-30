"""Dataset schemas for PA-LIME: protocol-aware feature metadata for NIDS datasets."""

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

    # Computed fields
    protocol_index: Optional[int] = field(init=False)
    non_negative_indices: list[int] = field(init=False)
    tcp_only_indices: list[int] = field(init=False)
    discrete_indices: list[int] = field(init=False)
    hierarchical_index_triples: list[tuple[int, int, int]] = field(init=False)

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

CIC_IDS_2017 = DatasetSchema(
    name="CIC-IDS-2017",
    feature_names=_CIC_2017_FEATURES,
    protocol_feature="Protocol",
    non_negative_features=_CIC_2017_NON_NEGATIVE,
    tcp_only_features=_CIC_2017_TCP_ONLY,
    discrete_features=_CIC_2017_DISCRETE,
    hierarchical_constraints=_CIC_2017_HIERARCHICAL,
    protocol_encoding="integer",
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

CSE_CIC_IDS2018 = DatasetSchema(
    name="CSE-CIC-IDS2018",
    feature_names=_CIC_2018_FEATURES,
    protocol_feature="Protocol",
    non_negative_features=_CIC_2018_NON_NEGATIVE,
    tcp_only_features=_CIC_2018_TCP_ONLY,
    discrete_features=_CIC_2018_DISCRETE,
    hierarchical_constraints=_CIC_2018_HIERARCHICAL,
    protocol_encoding="integer",
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

NSL_KDD = DatasetSchema(
    name="NSL-KDD",
    feature_names=_NSL_KDD_FEATURES,
    protocol_feature="protocol_type",
    non_negative_features=_NSL_KDD_NON_NEGATIVE,
    tcp_only_features=["urgent"],
    discrete_features=_NSL_KDD_DISCRETE,
    hierarchical_constraints=[],
    protocol_encoding="string",
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

UNSW_NB15_NATIVE = DatasetSchema(
    name="UNSW-NB15-Native",
    feature_names=_UNSW_NATIVE_FEATURES,
    protocol_feature="proto",
    non_negative_features=_UNSW_NATIVE_NON_NEGATIVE,
    tcp_only_features=_UNSW_NATIVE_TCP_ONLY,
    discrete_features=_UNSW_NATIVE_DISCRETE,
    hierarchical_constraints=[],
    protocol_encoding="string",
)


# ---------------------------------------------------------------------------
# UNSW-NB15 CICFlowMeter
# ---------------------------------------------------------------------------

_UNSW_CIC_FEATURES = [
    "Dst Port", "Protocol", "Flow Duration",
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
    "Dst Port", "Protocol",
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

UNSW_NB15_CIC = DatasetSchema(
    name="UNSW-NB15-CICFlowMeter",
    feature_names=_UNSW_CIC_FEATURES,
    protocol_feature="Protocol",
    non_negative_features=_UNSW_CIC_NON_NEGATIVE,
    tcp_only_features=_UNSW_CIC_TCP_ONLY,
    discrete_features=_UNSW_CIC_DISCRETE,
    hierarchical_constraints=_UNSW_CIC_HIERARCHICAL,
    protocol_encoding="integer",
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
