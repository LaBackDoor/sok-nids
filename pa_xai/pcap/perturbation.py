"""Packet-level header field perturbation (raw noise, pre-enforcement)."""

from __future__ import annotations

import numpy as np

from pa_xai.pcap.parser import ParsedPacket


class PacketPerturbator:
    """Applies raw Gaussian noise to mutable packet header fields.

    Does NOT enforce cross-field constraints — PacketConstraintEnforcer
    handles that after perturbation.
    """

    def perturb(self, packet: ParsedPacket, sigma: float, num_samples: int) -> list[ParsedPacket]:
        """Generate num_samples perturbed copies of a packet."""
        results = []
        for _ in range(num_samples):
            p = packet.copy()

            # IP fields (always perturbed)
            p.ip_ttl = int(round(packet.ip_ttl + np.random.normal(0, sigma)))
            p.ip_tos = int(round(packet.ip_tos + np.random.normal(0, sigma)))
            # ip_total_length NOT perturbed — recomputed by enforcer
            p.ip_flags = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7]))

            if packet.protocol == "tcp":
                p.tcp_window_size = int(round(packet.tcp_window_size + np.random.normal(0, sigma * 100)))
                p.tcp_flags = packet.tcp_flags ^ int(np.random.randint(0, 64))
                p.tcp_seq = int(round(packet.tcp_seq + np.random.normal(0, sigma * 1000)))
                p.tcp_ack = int(round(packet.tcp_ack + np.random.normal(0, sigma * 1000)))
                p.tcp_urgent_ptr = int(round(packet.tcp_urgent_ptr + np.random.normal(0, sigma)))
            elif packet.protocol == "icmp":
                p.icmp_type = int(np.random.randint(0, 256))
                p.icmp_code = int(np.random.randint(0, 256))
            # UDP: udp_length not perturbed (recomputed by enforcer)

            results.append(p)
        return results
