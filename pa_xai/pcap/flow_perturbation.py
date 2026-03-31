"""Flow-level perturbation: perturb packets then enforce flow constraints."""

from __future__ import annotations

from pa_xai.pcap.parser import ParsedFlow
from pa_xai.pcap.perturbation import PacketPerturbator
from pa_xai.pcap.packet_constraints import FlowConstraintEnforcer


class FlowPerturbator:
    """Perturb a flow by perturbing constituent packets then enforcing flow constraints."""

    def __init__(self, packet_perturbator: PacketPerturbator, flow_enforcer: FlowConstraintEnforcer) -> None:
        self.packet_perturbator = packet_perturbator
        self.flow_enforcer = flow_enforcer

    def perturb(self, flow: ParsedFlow, sigma: float, num_samples: int) -> list[ParsedFlow]:
        results = []
        for _ in range(num_samples):
            perturbed_packets = []
            for pkt in flow.packets:
                perturbed = self.packet_perturbator.perturb(pkt, sigma, num_samples=1)[0]
                perturbed_packets.append(perturbed)

            perturbed_flow = ParsedFlow(
                packets=perturbed_packets,
                protocol=flow.protocol,
                flow_key=flow.flow_key,
                pcap_path=None,
            )
            self.flow_enforcer.enforce(perturbed_flow, flow)
            results.append(perturbed_flow)
        return results
