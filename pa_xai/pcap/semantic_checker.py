"""Post-enforcement semantic validation for perturbed packets and flows."""

from __future__ import annotations

from pa_xai.pcap.parser import ParsedPacket, ParsedFlow
from pa_xai.pcap.packet_constraints import (
    VALID_ICMP_TYPES, VALID_ICMP_CODES,
    FIN, SYN, RST, URG,
    IP_HEADER_SIZE, UDP_HEADER_SIZE,
)


class SemanticChecker:
    """Final safety-net validator after constraint enforcement."""

    def check_packet(self, packet: ParsedPacket) -> bool:
        if packet.ip_ttl <= 0:
            return False
        if packet.ip_total_length < IP_HEADER_SIZE:
            return False

        # Protocol mutual exclusion
        if packet.protocol == "tcp":
            if packet.udp_length is not None or packet.icmp_type is not None:
                return False
        elif packet.protocol == "udp":
            if packet.tcp_flags is not None or packet.tcp_seq is not None or packet.tcp_window_size is not None:
                return False
            if packet.udp_length is not None and packet.udp_length < UDP_HEADER_SIZE:
                return False
        elif packet.protocol == "icmp":
            if packet.tcp_flags is not None or packet.udp_length is not None:
                return False

        # TCP flag legality
        if packet.protocol == "tcp" and packet.tcp_flags is not None:
            flags = packet.tcp_flags
            if (flags & SYN) and (flags & FIN) and (flags & RST):
                return False
            if (flags & SYN) and (flags & FIN):
                return False
            if (flags & SYN) and (flags & RST):
                return False
            if not (flags & URG) and packet.tcp_urgent_ptr != 0:
                return False
            if packet.tcp_seq < 0 or packet.tcp_seq > 2**32 - 1:
                return False
            if packet.tcp_ack < 0 or packet.tcp_ack > 2**32 - 1:
                return False

        # ICMP validity
        if packet.protocol == "icmp":
            if packet.icmp_type not in VALID_ICMP_TYPES:
                return False
            valid_codes = VALID_ICMP_CODES.get(packet.icmp_type, set())
            if packet.icmp_code not in valid_codes:
                return False

        return True

    def check_flow(self, flow: ParsedFlow) -> bool:
        if not flow.packets:
            return False
        for pkt in flow.packets:
            if not self.check_packet(pkt):
                return False
        for pkt in flow.packets:
            if pkt.protocol != flow.protocol:
                return False
        for i in range(1, len(flow.packets)):
            if flow.packets[i].timestamp < flow.packets[i - 1].timestamp:
                return False
        return True
