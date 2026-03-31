"""Packet and flow constraint enforcement for PCAP perturbation pipeline."""

from __future__ import annotations

from pa_xai.pcap.parser import ParsedPacket, ParsedFlow

# Valid ICMP types per IANA
VALID_ICMP_TYPES = {0, 3, 5, 8, 11, 12}

VALID_ICMP_CODES: dict[int, set[int]] = {
    0: {0},
    3: set(range(16)),
    5: {0, 1, 2, 3},
    8: {0},
    11: {0, 1},
    12: {0, 1, 2},
}

VALID_IP_FLAGS = {0, 2, 4}

# TCP flag bits
FIN = 0x01
SYN = 0x02
RST = 0x04
PSH = 0x08
ACK = 0x10
URG = 0x20

IP_HEADER_SIZE = 20
TCP_HEADER_SIZE = 20
UDP_HEADER_SIZE = 8
ICMP_HEADER_SIZE = 8


def _snap_to_nearest(value: int, valid_set: set[int]) -> int:
    return min(valid_set, key=lambda v: abs(v - value))


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


class PacketConstraintEnforcer:
    """7-step packet constraint enforcement:
    1. Pin protocol + protocol gating
    2. Pin identity fields
    3. Clamp to valid ranges
    4. TCP flag state repair
    5. Cross-field enforcement
    6. Discrete field rounding (handled in clamp via int(round()))
    7. Reconstruct raw packet (deferred to pipeline)
    """

    def enforce(self, packet: ParsedPacket, original: ParsedPacket) -> ParsedPacket:
        # 1. Pin protocol + gating
        packet.protocol = original.protocol
        if packet.protocol != "tcp":
            packet.tcp_window_size = None
            packet.tcp_flags = None
            packet.tcp_seq = None
            packet.tcp_ack = None
            packet.tcp_urgent_ptr = None
        if packet.protocol != "udp":
            packet.udp_length = None
        if packet.protocol != "icmp":
            packet.icmp_type = None
            packet.icmp_code = None

        # 2. Pin identity (IP/ports in raw_packet, not perturbed)

        # 3. Clamp fields to valid ranges
        packet.ip_ttl = _clamp(int(round(packet.ip_ttl)), 1, 255)
        packet.ip_tos = _clamp(int(round(packet.ip_tos)), 0, 255)
        packet.ip_flags = _snap_to_nearest(int(round(packet.ip_flags)), VALID_IP_FLAGS)

        if packet.protocol == "tcp":
            packet.tcp_window_size = _clamp(int(round(packet.tcp_window_size)), 0, 65535)
            packet.tcp_seq = _clamp(int(round(packet.tcp_seq)), 0, 2**32 - 1)
            packet.tcp_ack = _clamp(int(round(packet.tcp_ack)), 0, 2**32 - 1)
            packet.tcp_urgent_ptr = _clamp(int(round(packet.tcp_urgent_ptr)), 0, 65535)

        if packet.protocol == "icmp":
            packet.icmp_type = _snap_to_nearest(_clamp(int(round(packet.icmp_type)), 0, 255), VALID_ICMP_TYPES)
            valid_codes = VALID_ICMP_CODES.get(packet.icmp_type, {0})
            packet.icmp_code = _snap_to_nearest(_clamp(int(round(packet.icmp_code)), 0, 255), valid_codes)

        # 4. TCP flag state repair
        if packet.protocol == "tcp":
            packet.tcp_flags = self._repair_tcp_flags(packet.tcp_flags, original.tcp_flags)

        # 5. Cross-field enforcement
        if packet.protocol == "tcp":
            if not (packet.tcp_flags & URG):
                packet.tcp_urgent_ptr = 0
            packet.ip_total_length = IP_HEADER_SIZE + TCP_HEADER_SIZE + packet.payload_size
        elif packet.protocol == "udp":
            packet.udp_length = UDP_HEADER_SIZE + packet.payload_size
            packet.ip_total_length = IP_HEADER_SIZE + UDP_HEADER_SIZE + packet.payload_size
        elif packet.protocol == "icmp":
            packet.ip_total_length = IP_HEADER_SIZE + ICMP_HEADER_SIZE + packet.payload_size

        # 6. Discrete rounding already done via int(round()) in step 3
        # 7. Raw packet reconstruction deferred to pipeline

        return packet

    def _repair_tcp_flags(self, flags: int, original_flags: int) -> int:
        orig_is_syn = bool(original_flags & SYN) and not bool(original_flags & ACK)
        orig_is_synack = bool(original_flags & SYN) and bool(original_flags & ACK)
        orig_is_rst = bool(original_flags & RST)
        orig_is_fin = bool(original_flags & FIN)

        if orig_is_syn:
            flags = (flags | SYN) & ~FIN & ~RST & ~ACK
        elif orig_is_synack:
            flags = (flags | SYN | ACK) & ~FIN & ~RST
        elif orig_is_rst:
            flags = (flags | RST) & ~SYN & ~FIN
            flags = flags & (RST | ACK)
        elif orig_is_fin:
            flags = (flags | FIN | ACK) & ~RST & ~SYN
        else:
            # Established: ACK set, SYN off
            flags = (flags | ACK) & ~SYN
            if (flags & FIN) and (flags & RST):
                flags = flags & ~RST

        # Final safety
        if (flags & SYN) and (flags & FIN):
            flags = flags & ~FIN
        if (flags & SYN) and (flags & RST):
            flags = flags & ~RST
        if flags == 0:
            flags = original_flags

        return flags


class FlowConstraintEnforcer:
    """5-step flow constraint enforcement:
    1. Enforce each packet individually
    2. Pin protocol homogeneity
    3. Enforce temporal ordering
    4. TCP sequence repair
    5. Reconstruct flow PCAP (deferred to pipeline)
    """

    def __init__(self, packet_enforcer: PacketConstraintEnforcer):
        self.packet_enforcer = packet_enforcer

    def enforce(self, flow: ParsedFlow, original: ParsedFlow) -> ParsedFlow:
        # 1. Enforce each packet
        for i, pkt in enumerate(flow.packets):
            orig_pkt = original.packets[i] if i < len(original.packets) else original.packets[-1]
            self.packet_enforcer.enforce(pkt, orig_pkt)

        # 2. Pin protocol homogeneity
        for pkt in flow.packets:
            pkt.protocol = flow.protocol

        # 3. Enforce temporal ordering
        flow.packets.sort(key=lambda p: p.timestamp)

        # 4. TCP sequence repair
        if flow.protocol == "tcp":
            self._repair_tcp_sequences(flow)

        return flow

    def _repair_tcp_sequences(self, flow: ParsedFlow) -> None:
        if not flow.packets:
            return
        current_seq = flow.packets[0].tcp_seq
        for pkt in flow.packets:
            pkt.tcp_seq = current_seq
            current_seq += max(pkt.payload_size, 1)
