# tests/test_pa_xai/test_pcap/test_parser.py

def test_parsed_packet_tcp_fields():
    from pa_xai.pcap.parser import ParsedPacket

    pkt = ParsedPacket(
        raw_packet=None, protocol="tcp",
        ip_ttl=64, ip_tos=0, ip_total_length=60, ip_flags=2,
        tcp_window_size=65535, tcp_flags=0x02, tcp_seq=1000, tcp_ack=0,
        tcp_urgent_ptr=0, udp_length=None, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=0,
    )
    assert pkt.protocol == "tcp"
    assert pkt.tcp_flags == 0x02
    assert pkt.udp_length is None


def test_parsed_packet_udp_fields():
    from pa_xai.pcap.parser import ParsedPacket

    pkt = ParsedPacket(
        raw_packet=None, protocol="udp",
        ip_ttl=128, ip_tos=0, ip_total_length=60, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=40, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=32,
    )
    assert pkt.protocol == "udp"
    assert pkt.udp_length == 40
    assert pkt.tcp_flags is None


def test_parsed_packet_copy():
    from pa_xai.pcap.parser import ParsedPacket

    pkt = ParsedPacket(
        raw_packet=None, protocol="tcp",
        ip_ttl=64, ip_tos=0, ip_total_length=60, ip_flags=2,
        tcp_window_size=65535, tcp_flags=0x02, tcp_seq=1000, tcp_ack=0,
        tcp_urgent_ptr=0, udp_length=None, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=0,
    )
    c = pkt.copy()
    c.ip_ttl = 128
    assert pkt.ip_ttl == 64  # original unchanged
    assert c.ip_ttl == 128


def test_parsed_flow_contains_packets():
    from pa_xai.pcap.parser import ParsedPacket, ParsedFlow

    pkt1 = ParsedPacket(
        raw_packet=None, protocol="tcp",
        ip_ttl=64, ip_tos=0, ip_total_length=60, ip_flags=2,
        tcp_window_size=65535, tcp_flags=0x02, tcp_seq=1000, tcp_ack=0,
        tcp_urgent_ptr=0, udp_length=None, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=0,
    )
    pkt2 = ParsedPacket(
        raw_packet=None, protocol="tcp",
        ip_ttl=64, ip_tos=0, ip_total_length=60, ip_flags=2,
        tcp_window_size=65535, tcp_flags=0x12, tcp_seq=5000, tcp_ack=1001,
        tcp_urgent_ptr=0, udp_length=None, icmp_type=None, icmp_code=None,
        timestamp=1000.001, payload_size=0,
    )
    flow = ParsedFlow(
        packets=[pkt1, pkt2], protocol="tcp",
        flow_key=("10.0.0.1", "10.0.0.2", 12345, 80, "tcp"),
        pcap_path=None,
    )
    assert len(flow.packets) == 2
    assert flow.protocol == "tcp"


def test_pcap_parser_with_real_pcap():
    """Integration test: create a PCAP with stackforge, parse it back."""
    import tempfile, os
    import stackforge
    from pa_xai.pcap.parser import PcapParser

    # Build a TCP SYN packet
    pkt = stackforge.Ether()/stackforge.IP(
        dst='10.0.0.1', src='10.0.0.2', ttl=64
    )/stackforge.TCP(dport=80, sport=12345, flags='S', window=65535, seq=1000, ack=0)

    tmp = tempfile.mktemp(suffix='.pcap')
    try:
        stackforge.wrpcap(tmp, [pkt])
        parser = PcapParser()
        packets = parser.parse_packets(tmp)
        assert len(packets) == 1
        p = packets[0]
        assert p.protocol == "tcp"
        assert p.ip_ttl == 64
        assert p.tcp_flags == 0x02  # SYN
        assert p.tcp_seq == 1000
        assert p.tcp_window_size == 65535
    finally:
        os.unlink(tmp)


def test_pcap_parser_udp():
    """Integration test with a UDP packet."""
    import tempfile, os
    import stackforge
    from pa_xai.pcap.parser import PcapParser

    pkt = stackforge.Ether()/stackforge.IP(
        dst='10.0.0.1', src='10.0.0.2', ttl=128
    )/stackforge.UDP(dport=53, sport=5000)

    tmp = tempfile.mktemp(suffix='.pcap')
    try:
        stackforge.wrpcap(tmp, [pkt])
        parser = PcapParser()
        packets = parser.parse_packets(tmp)
        assert len(packets) == 1
        p = packets[0]
        assert p.protocol == "udp"
        assert p.ip_ttl == 128
        assert p.tcp_flags is None
        assert p.udp_length is not None
    finally:
        os.unlink(tmp)
