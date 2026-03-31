# PA-XAI: Constraint Hardening & PCAP Pipeline Design

**Date:** 2026-03-30
**Goal:** Extend PA-XAI with (A) 6 missing CSV-mode constraints for precise, semantically valid perturbations, and (B) a full PCAP pipeline using stackforge for packet-level and flow-level model explanation with semantic checking.

**Scope:** All four explainers (LIME, IG, DeepLIFT, SHAP) gain both the hardened constraints and PCAP support.

**Approach:** Layered Constraints (Approach C) — `core/constraints.py` handles numpy-array constraints for CSV mode; `pcap/packet_constraints.py` handles packet-object validation using stackforge. An adapter bridges them when flow-level PCAP models need both.

---

## Architecture

```
pa_xai/
├── core/
│   ├── schemas.py              # + BoundedRangeConstraint, CrossFeatureConstraint,
│   │                           #   StdRangeConstraint, binary_features, rate_features,
│   │                           #   udp_only_features, icmp_only_features, duplicate_features
│   ├── constraints.py          # + 6 new constraint enforcement steps (11 total)
│   ├── result.py               # unchanged
│   ├── baseline.py             # unchanged
│   └── metrics.py              # unchanged
├── pcap/
│   ├── __init__.py
│   ├── parser.py               # Stackforge PCAP reader → ParsedPacket / ParsedFlow
│   ├── packet_constraints.py   # Protocol-header semantic checks
│   ├── perturbation.py         # Packet-level header field perturbation
│   ├── flow_perturbation.py    # Flow-level: perturb packets, re-extract via stackforge
│   ├── semantic_checker.py     # Hard-reject validator (packet + flow cross-validation)
│   └── pipeline.py             # Orchestrator: parse → perturb → check → reject/regen
├── lime/
│   ├── explainer.py            # + explain_pcap() method
│   ├── fuzzer.py               # unchanged (uses enhanced ConstraintEnforcer automatically)
│   └── metrics.py              # + semantic_robustness_pcap()
├── ig/
│   ├── explainer.py            # + explain_pcap() method
│   └── metrics.py              # unchanged
├── deeplift/
│   ├── explainer.py            # + explain_pcap() method
│   └── metrics.py              # unchanged
└── shap/
    ├── explainer.py            # + explain_pcap() method
    └── metrics.py              # unchanged
```

---

## Workstream A: CSV-Mode Constraint Hardening

### New Schema Types

```python
@dataclass(frozen=True)
class BoundedRangeConstraint:
    feature: str
    lower: float
    upper: float

@dataclass(frozen=True)
class CrossFeatureConstraint:
    derived_feature: str
    relation: str         # "ratio" (a/b), "sum_ratio" ((a+b)/c), "square" (a^2), "equal" (a=b)
    operands: list[str]   # ["numerator", "denominator"] or ["a", "b", "denominator"] for sum_ratio

@dataclass(frozen=True)
class StdRangeConstraint:
    std_feature: str
    max_feature: str
    min_feature: str
```

### New DatasetSchema Fields

```python
# Added to DatasetSchema init fields:
bounded_range_constraints: list[BoundedRangeConstraint]
cross_feature_constraints: list[CrossFeatureConstraint]
std_range_constraints: list[StdRangeConstraint]
udp_only_features: list[str]
icmp_only_features: list[str]
duplicate_features: list[tuple[str, str]]

# Computed in __post_init__:
bounded_range_index_bounds: list[tuple[int, float, float]]
cross_feature_index_triples: list[tuple[int, int, int, str]]
std_range_index_triples: list[tuple[int, int, int]]
udp_only_indices: list[int]
icmp_only_indices: list[int]
duplicate_index_pairs: list[tuple[int, int]]
```

### Constraint 6: Bounded Range Clamping

Clamps features to their physically valid range.

Per-schema definitions:

**All CIC schemas (2017, 2018, UNSW-CIC):**
- Port features: [0, 65535]
- Init window bytes: [0, 65535]

**NSL-KDD:**
- All `*_rate` features (serror_rate, srv_serror_rate, rerror_rate, srv_rerror_rate, same_srv_rate, diff_srv_rate, srv_diff_host_rate, dst_host_same_srv_rate, dst_host_diff_srv_rate, dst_host_same_src_port_rate, dst_host_srv_diff_host_rate, dst_host_serror_rate, dst_host_srv_serror_rate, dst_host_rerror_rate, dst_host_srv_rerror_rate): [0.0, 1.0]
- Binary features (land, logged_in, root_shell, su_attempted, is_host_login, is_guest_login): [0, 1]

**UNSW-NB15 Native:**
- TTL (sttl, dttl): [0, 255]
- Window (swin, dwin): [0, 65535]
- Port features (Srcport, Dstport): [0, 65535]
- Binary (is_ftp_login, is_sm_ips_ports): [0, 1]

Enforcement: `neighborhood[:, idx] = np.clip(neighborhood[:, idx], lower, upper)`

### Constraint 7: Cross-Feature Arithmetic

Recomputes derived features from their operands after perturbation.

**CIC-IDS-2017 / CSE-CIC-IDS2018 / UNSW-CIC constraints:**

| Derived | Relation | Operand A | Operand B |
|---|---|---|---|
| Flow Bytes/s | ratio | Total Length Fwd + Bwd Packets (sum) | Flow Duration |
| Flow Packets/s | ratio | Total Fwd + Bwd Packets (sum) | Flow Duration |
| Fwd Packets/s | ratio | Total Fwd Packets | Flow Duration |
| Bwd Packets/s | ratio | Total Backward Packets | Flow Duration |
| Packet Length Variance | square | Packet Length Std | — |
| Subflow Fwd Packets | equal | Total Fwd Packets | — |
| Subflow Fwd Bytes | equal | Total Length of Fwd Packets | — |
| Subflow Bwd Packets | equal | Total Backward Packets | — |
| Subflow Bwd Bytes | equal | Total Length of Bwd Packets | — |

Note: For ratio-type constraints, the derived feature is recomputed as `operands[0] / operands[1]`. When the denominator is 0, the derived feature is set to 0. Sum-ratio constraints use `(operands[0] + operands[1]) / operands[2]`, with the same zero-denominator rule.

Specific sum_ratio constraints:
- `Flow Bytes/s`: sum_ratio of [Total Length Fwd, Total Length Bwd, Flow Duration]
- `Flow Packets/s`: sum_ratio of [Total Fwd Packets, Total Backward Packets, Flow Duration]

Enforcement order: recompute derived from operands. For "equal", copy operand_a to derived. For "square", derived = operand_a^2. For "ratio", derived = a / b (0 if b == 0). For "sum_ratio", derived = (a + b) / c (0 if c == 0).

### Constraint 8: Std <= Range

For every hierarchical triple that has an associated std feature, enforce `std <= max - min`.

**CIC-IDS-2017 std constraints:**
- (Fwd Packet Length Std, Fwd Packet Length Max, Fwd Packet Length Min)
- (Bwd Packet Length Std, Bwd Packet Length Max, Bwd Packet Length Min)
- (Packet Length Std, Max Packet Length, Min Packet Length)
- (Flow IAT Std, Flow IAT Max, Flow IAT Min)
- (Fwd IAT Std, Fwd IAT Max, Fwd IAT Min)
- (Bwd IAT Std, Bwd IAT Max, Bwd IAT Min)
- (Active Std, Active Max, Active Min)
- (Idle Std, Idle Max, Idle Min)

Same pattern for CSE-CIC-IDS2018 and UNSW-CIC with their respective feature names.

Enforcement: `neighborhood[:, std_i] = np.minimum(neighborhood[:, std_i], neighborhood[:, max_i] - neighborhood[:, min_i])`

### Constraint 9: UDP/ICMP-Specific Gating

Zeroes features that are meaningless for non-matching protocols.

**NSL-KDD:**
- `icmp_only_features`: [] (no ICMP-specific features in NSL-KDD)
- `udp_only_features`: [] (no UDP-specific features in NSL-KDD)
- Features zeroed for ICMP: `num_failed_logins`, `logged_in`, `root_shell`, `su_attempted`, `num_shells`, `num_access_files` (connection/session features that don't apply to ICMP). These go into a broader `connection_session_features` list that is zeroed when protocol == ICMP.

**UNSW-NB15 Native:**
- `icmp_only_features`: []
- Features zeroed for ICMP: `trans_depth`, `response_body_len`, `is_ftp_login`, `ct_ftp_cmd`, `ct_flw_http_mthd` (application-layer features impossible for ICMP)

Enforcement: Same pattern as TCP-gating. Check protocol encoding, determine if current protocol is UDP or ICMP, zero the relevant indices.

For integer encoding: UDP = 17, ICMP = 1.
For string encoding: values are dataset-specific (depends on label encoder). Alphabetical label encoding gives: ICMP=0, TCP=1, UDP=2 for NSL-KDD and UNSW-NB15 Native (alphabetical order: icmp < tcp < udp). The user can override via `tcp_label_value`, `udp_label_value`, and `icmp_label_value` parameters. Defaults assume alphabetical encoding: TCP=0, UDP=2, ICMP=1. Note: existing code assumes TCP=0 for string encoding, which matches datasets where tcp sorts first — but the correct NSL-KDD alphabetical order is icmp=0, tcp=1, udp=2. The defaults will be updated to match the actual alphabetical sort.

Updated: add `udp_label_value` and `icmp_label_value` parameters to `ConstraintEnforcer.enforce()` and `DomainConstraintFuzzer.__init__()`.

### Constraint 10: Duplicate Feature Equality

**CIC-IDS-2017:** `("Fwd Header Length", "Fwd Header Length.1")` — these are duplicate columns.

Enforcement: `neighborhood[:, idx_b] = neighborhood[:, idx_a]` — the second feature is always set equal to the first.

### Full Enforcement Order (11 Steps)

1. Fix protocol column to original value
2. Non-negativity clamping
3. Hierarchical ordering (max >= mean >= min)
4. Std <= range (std <= max - min)
5. Cross-feature arithmetic recomputation
6. Bounded range clamping
7. Discrete rounding
8. TCP-only feature zeroing
9. UDP-specific feature zeroing
10. ICMP-specific feature zeroing
11. Duplicate feature equality

Rationale for order:
- Steps 1-2: Establish basic validity (protocol fixed, no negatives)
- Steps 3-4: Statistical consistency (hierarchies, then std bounded by range)
- Step 5: Arithmetic consistency (derived features recomputed from valid operands)
- Step 6: Physical bounds (after arithmetic so recomputed values get clamped too)
- Step 7: Integer integrity (after clamping so discrete values are in valid range)
- Steps 8-10: Protocol gating (after all other constraints, zeroing overrides everything)
- Step 11: Duplicate equality (last, so duplicates match final state)

---

## Workstream B: PCAP Pipeline

### Data Model (`pcap/parser.py`)

```python
@dataclass
class ParsedPacket:
    """Single packet with mutable header fields for perturbation."""
    raw_packet: Any             # stackforge packet object
    protocol: str               # "tcp", "udp", "icmp"
    # IP fields
    ip_ttl: int
    ip_tos: int
    ip_total_length: int
    ip_flags: int
    # TCP fields (None for non-TCP)
    tcp_window_size: int | None
    tcp_flags: int | None       # bitmask: FIN=0x01, SYN=0x02, RST=0x04, PSH=0x08, ACK=0x10, URG=0x20
    tcp_seq: int | None
    tcp_ack: int | None
    tcp_urgent_ptr: int | None
    # UDP fields (None for non-UDP)
    udp_length: int | None
    # ICMP fields (None for non-ICMP)
    icmp_type: int | None
    icmp_code: int | None
    # Metadata
    timestamp: float
    payload_size: int

@dataclass
class ParsedFlow:
    """Bidirectional flow as a sequence of packets."""
    packets: list[ParsedPacket]
    protocol: str               # dominant protocol of the flow
    flow_key: tuple             # (src_ip, dst_ip, src_port, dst_port, proto)
    pcap_path: str | None       # source PCAP if available
```

### Parser (`pcap/parser.py`)

```python
class PcapParser:
    def parse_packets(self, pcap_path: str) -> list[ParsedPacket]:
        """Read PCAP using stackforge.rdpcap(), extract header fields into ParsedPacket."""

    def parse_flows(self, pcap_path: str) -> list[ParsedFlow]:
        """Read PCAP, extract bidirectional flows using stackforge flow extraction,
        return list of ParsedFlow each containing its constituent ParsedPackets."""
```

### Packet-Level Perturbation (`pcap/perturbation.py`)

```python
class PacketPerturbator:
    def perturb(self, packet: ParsedPacket, sigma: float, num_samples: int) -> list[ParsedPacket]:
        """Generate num_samples perturbed copies of a single packet (header fields only)."""
```

Field-specific perturbation strategies:

| Field | Strategy | Range |
|---|---|---|
| ip_ttl | Gaussian + round + clip | [1, 255] |
| ip_tos | Gaussian + round + clip | [0, 255] |
| ip_total_length | Pinned (recomputed from headers) | — |
| ip_flags | Uniform from valid set {0, 2, 4} | discrete |
| tcp_window_size | Gaussian + round + clip | [0, 65535] |
| tcp_flags | Bit-flip with protocol-aware rules | bitmask |
| tcp_seq / tcp_ack | Gaussian + round + clip | [0, 2^32 - 1] |
| tcp_urgent_ptr | Zero unless URG flag set | [0, 65535] |
| udp_length | Pinned (recomputed) | — |
| icmp_type | Uniform from valid types {0,3,5,8,11,12} | discrete |
| icmp_code | Uniform from valid codes for type | discrete |

TCP flag perturbation rules:
- SYN packets: SYN set, FIN and RST off
- SYN-ACK: SYN+ACK set, FIN and RST off
- Established: ACK set, SYN off
- FIN/RST: exactly one, never both
- PSH/URG: freely toggled when ACK set

### Flow-Level Perturbation (`pcap/flow_perturbation.py`)

```python
class FlowPerturbator:
    def perturb(self, flow: ParsedFlow, sigma: float, num_samples: int) -> list[ParsedFlow]:
        """Perturb each packet in the flow independently using PacketPerturbator,
        write perturbed packets to temp PCAP via stackforge.wrpcap(),
        re-extract flow via stackforge to guarantee valid flow structure."""
```

### Semantic Checker (`pcap/semantic_checker.py`)

Hard-reject validation. A sample must pass ALL checks or be discarded.

```python
class SemanticChecker:
    def check_packet(self, packet: ParsedPacket) -> bool:
        """Validate a single perturbed packet."""

    def check_flow(self, flow: ParsedFlow) -> bool:
        """Validate a perturbed flow (runs check_packet on each, plus flow-level checks)."""
```

**Packet-level checks:**
1. IP header: `ip_total_length >= 20`, total_length matches header + payload
2. TCP flag legality: no SYN+FIN+RST simultaneously, URG ptr = 0 when URG off, seq/ack within 32-bit
3. TCP state coherence: SYN packets have no payload (or <= MSS), RST minimal payload
4. UDP: `udp_length >= 8`
5. ICMP: only valid (type, code) pairs per IANA registry
6. TTL: > 0
7. Protocol mutual exclusion: TCP fields None for UDP/ICMP, vice versa

**Flow-level cross-validation checks:**
1. Protocol homogeneity: all packets share L4 protocol
2. Temporal ordering: timestamps monotonically non-decreasing
3. TCP sequence consistency: seq numbers advance by payload size within window
4. Byte count: sum of payload sizes matches expected total
5. Packet count: len(packets) == expected
6. Bidirectionality: at least one packet per direction for TCP with handshake

### Pipeline Orchestrator (`pcap/pipeline.py`)

```python
class PcapPipeline:
    def __init__(
        self,
        parser: PcapParser | None = None,
        packet_perturbator: PacketPerturbator | None = None,
        flow_perturbator: FlowPerturbator | None = None,
        checker: SemanticChecker | None = None,
        max_retries: int = 10,
    ):
        """Defaults created if None provided."""

    def generate_neighborhood(
        self,
        pcap_path: str,
        num_samples: int,
        sigma: float,
        mode: str = "packet",   # "packet" or "flow"
    ) -> list[ParsedPacket] | list[ParsedFlow]:
        """Generate num_samples semantically valid perturbed samples.

        Generates in batches. Each batch: perturb → check → keep valid.
        Accumulates until num_samples reached. Raises RuntimeError if
        max_retries * num_samples total attempts exhausted.
        """
```

---

## Explainer PCAP Adapters

All four explainers gain an `explain_pcap()` method. The user provides:
- `predict_fn`: model prediction callable over packets/flows
- `feature_fn`: extracts numeric feature vector from ParsedPacket or ParsedFlow

PA-XAI handles perturbation, semantic validity, and explanation generation.

### LIME

```python
def explain_pcap(
    self,
    pcap_path: str,
    predict_fn,         # list[ParsedPacket|ParsedFlow] -> np.ndarray
    feature_fn,         # ParsedPacket|ParsedFlow -> np.ndarray (1D feature vector)
    mode: str = "packet",
    num_samples: int = 5000,
    sigma: float = 0.1,
    kernel_width: float | None = None,
    class_to_explain: int | None = None,
    max_retries: int = 10,
) -> ExplanationResult:
```

Pipeline:
1. `PcapPipeline.generate_neighborhood()` → list of valid perturbed samples
2. Extract features via `feature_fn` for each sample → neighborhood matrix
3. Get predictions via `predict_fn` → y_neighborhood
4. Compute distances, weights, fit Ridge surrogate (same as CSV mode)
5. Return `ExplanationResult`

Feature names: derived from `feature_fn` — user provides a `feature_names: list[str]` argument or the feature_fn returns a named structure.

### IG

```python
def explain_pcap(
    self,
    pcap_path: str,
    feature_fn,         # ParsedPacket|ParsedFlow -> np.ndarray
    mode: str = "packet",
    target: int | None = None,
    n_steps: int = 50,
) -> ExplanationResult:
```

Pipeline:
1. Parse PCAP → extract features via `feature_fn` → input tensor
2. Baseline via `get_protocol_valid_baseline()` on feature space
3. Integration path in feature space with `core/constraints.py` enforcement at each step
4. Same constrained IG logic as CSV mode

### DeepLIFT

```python
def explain_pcap(
    self,
    pcap_path: str,
    feature_fn,
    mode: str = "packet",
    target: int | None = None,
) -> ExplanationResult:
```

Pipeline:
1. Parse PCAP → extract features → input tensor
2. Baseline via `get_protocol_valid_baseline()`
3. Run existing DeepLIFT on feature vectors

### SHAP

```python
def explain_pcap(
    self,
    pcap_path: str,
    predict_fn,
    feature_fn,
    mode: str = "packet",
    target: int | None = None,
    nsamples: int | str = "auto",
) -> ExplanationResult:
```

Pipeline (kernel backend):
1. `PcapPipeline` generates background set from semantically valid perturbations
2. Constrained predict wraps `feature_fn` + `predict_fn`
3. Standard kernel SHAP on feature space

Pipeline (deep/tree backend):
1. Parse PCAP → extract features
2. Use existing deep/tree SHAP logic on feature vectors

---

## Metrics

### Existing (unchanged)
- `core/metrics.py`: `sparsity()` — works on any `ExplanationResult`
- `lime/metrics.py`: `fidelity()` — reads `r_squared`
- `lime/metrics.py`: `semantic_robustness()` — CSV mode, unchanged
- `ig/metrics.py`: `path_convergence()` — reads `convergence_delta`
- `deeplift/metrics.py`: `convergence_delta()` — reads `convergence_delta`
- `shap/metrics.py`: `additivity_check()` — reads `expected_value`

### New
- `lime/metrics.py`: `semantic_robustness_pcap()` — same as `semantic_robustness()` but mutations come from `PcapPipeline` instead of numpy perturbations

```python
def semantic_robustness_pcap(
    pcap_path: str,
    explainer: ProtocolAwareLIME,
    predict_fn,
    feature_fn,
    mode: str = "packet",
    epsilon: float = 0.05,
    n_iter: int = 50,
    num_samples: int = 5000,
    sigma: float = 0.1,
) -> float:
    """Mean Spearman rank correlation across n_iter semantically valid PCAP mutations."""
```

---

## Dependencies

- `stackforge >= 0.7.3` (already in pyproject.toml)
- No new dependencies required

## Key Design Decisions

1. **Two separate constraint systems**: numpy-array constraints in `core/` for CSV mode, packet-object validation in `pcap/` for PCAP mode. No forced abstraction between fundamentally different data types.
2. **Hard-reject for PCAP**: perturbed samples that fail semantic checks are discarded and regenerated, guaranteeing 100% valid neighborhoods.
3. **User provides feature_fn**: PA-XAI is model-agnostic. The user extracts features from packets/flows however their model expects. PA-XAI handles perturbation and validity.
4. **Stackforge for flow reconstruction**: perturbed packets are written back as PCAPs, stackforge re-extracts flows, guaranteeing valid flow structure.
5. **11-step enforcement order** for CSV constraints: protocol fix → non-negativity → hierarchical → std-range → cross-feature arithmetic → bounded range → discrete rounding → TCP zeroing → UDP zeroing → ICMP zeroing → duplicate equality.
