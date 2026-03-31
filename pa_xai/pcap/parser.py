"""Stackforge PCAP reader — parses PCAPs into ParsedPacket and ParsedFlow objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedPacket:
    """Single packet with mutable header fields for perturbation."""
    raw_packet: Any             # stackforge Packet object
    protocol: str               # "tcp", "udp", "icmp"
    ip_ttl: int
    ip_tos: int
    ip_total_length: int
    ip_flags: int
    tcp_window_size: int | None
    tcp_flags: int | None
    tcp_seq: int | None
    tcp_ack: int | None
    tcp_urgent_ptr: int | None
    udp_length: int | None
    icmp_type: int | None
    icmp_code: int | None
    timestamp: float
    payload_size: int

    def copy(self) -> ParsedPacket:
        """Create a deep copy for perturbation."""
        return ParsedPacket(
            raw_packet=copy.deepcopy(self.raw_packet),
            protocol=self.protocol,
            ip_ttl=self.ip_ttl,
            ip_tos=self.ip_tos,
            ip_total_length=self.ip_total_length,
            ip_flags=self.ip_flags,
            tcp_window_size=self.tcp_window_size,
            tcp_flags=self.tcp_flags,
            tcp_seq=self.tcp_seq,
            tcp_ack=self.tcp_ack,
            tcp_urgent_ptr=self.tcp_urgent_ptr,
            udp_length=self.udp_length,
            icmp_type=self.icmp_type,
            icmp_code=self.icmp_code,
            timestamp=self.timestamp,
            payload_size=self.payload_size,
        )


@dataclass
class ParsedFlow:
    """Bidirectional flow as a sequence of packets."""
    packets: list[ParsedPacket]
    protocol: str
    flow_key: tuple
    pcap_path: str | None


class PcapParser:
    """Read a PCAP file using stackforge and produce ParsedPacket/ParsedFlow objects."""

    def parse_packets(self, pcap_path: str) -> list[ParsedPacket]:
        """Read PCAP, return list of ParsedPacket (one per IP packet)."""
        import stackforge
        LK = stackforge.LayerKind

        raw_packets = stackforge.rdpcap(pcap_path)
        result = []
        for pcap_pkt in raw_packets:
            inner = pcap_pkt.packet
            if not inner.has_layer(LK.Ipv4):
                continue

            timestamp = float(pcap_pkt.time)
            ip_ttl = int(inner.getfieldval(LK.Ipv4, 'ttl'))
            ip_tos = int(inner.getfieldval(LK.Ipv4, 'tos'))
            ip_total_length = int(inner.getfieldval(LK.Ipv4, 'len'))
            ip_flags = int(inner.getfieldval(LK.Ipv4, 'flags'))

            protocol = "unknown"
            tcp_fields = dict(tcp_window_size=None, tcp_flags=None, tcp_seq=None,
                              tcp_ack=None, tcp_urgent_ptr=None)
            udp_length = None
            icmp_type = None
            icmp_code = None
            payload_size = 0

            if inner.has_layer(LK.Tcp):
                protocol = "tcp"
                tcp_fields = dict(
                    tcp_window_size=int(inner.getfieldval(LK.Tcp, 'window')),
                    tcp_flags=int(inner.getfieldval(LK.Tcp, 'flags')),
                    tcp_seq=int(inner.getfieldval(LK.Tcp, 'seq')),
                    tcp_ack=int(inner.getfieldval(LK.Tcp, 'ack')),
                    tcp_urgent_ptr=int(inner.getfieldval(LK.Tcp, 'urgptr')),
                )
                # Payload = IP total length - IP header (20) - TCP header (20)
                payload_size = max(0, ip_total_length - 40)
            elif inner.has_layer(LK.Udp):
                protocol = "udp"
                udp_length = int(inner.getfieldval(LK.Udp, 'len'))
                # Payload = UDP length - UDP header (8)
                payload_size = max(0, udp_length - 8)
            elif inner.has_layer(LK.Icmp):
                protocol = "icmp"
                icmp_type = int(inner.getfieldval(LK.Icmp, 'type'))
                icmp_code = int(inner.getfieldval(LK.Icmp, 'code'))
                # Payload = IP total length - IP header (20) - ICMP header (8)
                payload_size = max(0, ip_total_length - 28)
            else:
                continue  # Skip non-TCP/UDP/ICMP

            result.append(ParsedPacket(
                raw_packet=inner,
                protocol=protocol,
                ip_ttl=ip_ttl,
                ip_tos=ip_tos,
                ip_total_length=ip_total_length,
                ip_flags=ip_flags,
                **tcp_fields,
                udp_length=udp_length,
                icmp_type=icmp_type,
                icmp_code=icmp_code,
                timestamp=timestamp,
                payload_size=payload_size,
            ))
        return result

    def parse_flows(self, pcap_path: str) -> list[ParsedFlow]:
        """Read PCAP, extract bidirectional flows, return list of ParsedFlow."""
        import stackforge

        packets = self.parse_packets(pcap_path)
        convos = stackforge.extract_flows(pcap_path)

        result = []
        for conv in convos:
            proto = conv.protocol.lower()
            flow_pkts = []
            for idx in conv.packet_indices:
                if idx < len(packets):
                    flow_pkts.append(packets[idx])
            if not flow_pkts:
                continue

            flow_key = (conv.src_addr, conv.dst_addr, conv.src_port, conv.dst_port, proto)
            flow_pkts.sort(key=lambda p: p.timestamp)

            result.append(ParsedFlow(
                packets=flow_pkts,
                protocol=proto,
                flow_key=flow_key,
                pcap_path=pcap_path,
            ))
        return result
