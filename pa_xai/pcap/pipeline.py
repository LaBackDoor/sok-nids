"""Orchestrator: parse -> perturb -> enforce -> check -> accept."""

from __future__ import annotations

from pa_xai.pcap.parser import PcapParser, ParsedPacket, ParsedFlow
from pa_xai.pcap.perturbation import PacketPerturbator
from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer, FlowConstraintEnforcer
from pa_xai.pcap.flow_perturbation import FlowPerturbator
from pa_xai.pcap.semantic_checker import SemanticChecker


class PcapPipeline:
    """Generates semantically valid perturbed neighborhoods from PCAP data."""

    def __init__(
        self,
        parser: PcapParser | None = None,
        packet_perturbator: PacketPerturbator | None = None,
        packet_enforcer: PacketConstraintEnforcer | None = None,
        flow_perturbator: FlowPerturbator | None = None,
        flow_enforcer: FlowConstraintEnforcer | None = None,
        checker: SemanticChecker | None = None,
        max_retries: int = 10,
    ) -> None:
        self.parser = parser or PcapParser()
        self.packet_perturbator = packet_perturbator or PacketPerturbator()
        self.packet_enforcer = packet_enforcer or PacketConstraintEnforcer()
        if flow_enforcer is None:
            flow_enforcer = FlowConstraintEnforcer(self.packet_enforcer)
        self.flow_enforcer = flow_enforcer
        self.flow_perturbator = flow_perturbator or FlowPerturbator(
            self.packet_perturbator, self.flow_enforcer,
        )
        self.checker = checker or SemanticChecker()
        self.max_retries = max_retries

    def generate_neighborhood(
        self, pcap_path: str, num_samples: int, sigma: float, mode: str = "packet",
    ) -> list[ParsedPacket] | list[ParsedFlow]:
        if mode == "packet":
            packets = self.parser.parse_packets(pcap_path)
            return self.generate_neighborhood_from_packets(packets, num_samples, sigma)
        elif mode == "flow":
            flows = self.parser.parse_flows(pcap_path)
            if not flows:
                raise ValueError("No flows extracted from PCAP")
            return self.generate_neighborhood_from_flow(flows[0], num_samples, sigma)
        else:
            raise ValueError(f"mode must be 'packet' or 'flow', got {mode!r}")

    def generate_neighborhood_from_packets(
        self, packets: list[ParsedPacket], num_samples: int, sigma: float,
    ) -> list[ParsedPacket]:
        valid = []
        total_attempts = 0
        max_total = self.max_retries * num_samples

        while len(valid) < num_samples and total_attempts < max_total:
            remaining = num_samples - len(valid)
            for pkt in packets:
                batch = self.packet_perturbator.perturb(pkt, sigma, remaining)
                for p in batch:
                    self.packet_enforcer.enforce(p, pkt)
                    if self.checker.check_packet(p):
                        valid.append(p)
                        if len(valid) >= num_samples:
                            break
                if len(valid) >= num_samples:
                    break
            total_attempts += remaining

        if len(valid) < num_samples:
            raise RuntimeError(
                f"Could only generate {len(valid)}/{num_samples} valid samples "
                f"after {max_total} attempts"
            )
        return valid[:num_samples]

    def generate_neighborhood_from_flow(
        self, flow: ParsedFlow, num_samples: int, sigma: float,
    ) -> list[ParsedFlow]:
        valid = []
        total_attempts = 0
        max_total = self.max_retries * num_samples

        while len(valid) < num_samples and total_attempts < max_total:
            remaining = num_samples - len(valid)
            batch = self.flow_perturbator.perturb(flow, sigma, remaining)
            for f in batch:
                if self.checker.check_flow(f):
                    valid.append(f)
                    if len(valid) >= num_samples:
                        break
            total_attempts += remaining

        if len(valid) < num_samples:
            raise RuntimeError(
                f"Could only generate {len(valid)}/{num_samples} valid flows "
                f"after {max_total} attempts"
            )
        return valid[:num_samples]
