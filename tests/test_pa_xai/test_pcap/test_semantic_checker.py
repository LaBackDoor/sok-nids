from pa_xai.pcap.parser import ParsedPacket, ParsedFlow


def _make_tcp_packet(**overrides):
    defaults = dict(
        raw_packet=None, protocol="tcp",
        ip_ttl=64, ip_tos=0, ip_total_length=60, ip_flags=2,
        tcp_window_size=65535, tcp_flags=0x10, tcp_seq=1000, tcp_ack=500,
        tcp_urgent_ptr=0, udp_length=None, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=20,
    )
    defaults.update(overrides)
    return ParsedPacket(**defaults)


def test_valid_tcp_packet_passes():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    assert SemanticChecker().check_packet(_make_tcp_packet()) is True


def test_ttl_zero_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    assert SemanticChecker().check_packet(_make_tcp_packet(ip_ttl=0)) is False


def test_ip_total_length_too_small_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    assert SemanticChecker().check_packet(_make_tcp_packet(ip_total_length=10)) is False


def test_syn_fin_rst_simultaneously_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    assert SemanticChecker().check_packet(_make_tcp_packet(tcp_flags=0x07)) is False


def test_urg_ptr_nonzero_without_urg_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    assert SemanticChecker().check_packet(_make_tcp_packet(tcp_flags=0x10, tcp_urgent_ptr=100)) is False


def test_udp_length_too_small_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    pkt = ParsedPacket(
        raw_packet=None, protocol="udp", ip_ttl=64, ip_tos=0, ip_total_length=28, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=4, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=0,
    )
    assert SemanticChecker().check_packet(pkt) is False


def test_tcp_fields_on_udp_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    pkt = ParsedPacket(
        raw_packet=None, protocol="udp", ip_ttl=64, ip_tos=0, ip_total_length=28, ip_flags=0,
        tcp_window_size=65535, tcp_flags=0x10, tcp_seq=1000, tcp_ack=500,
        tcp_urgent_ptr=0, udp_length=20, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=12,
    )
    assert SemanticChecker().check_packet(pkt) is False


def test_valid_flow_passes():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    pkts = [_make_tcp_packet(timestamp=1.0), _make_tcp_packet(timestamp=2.0)]
    flow = ParsedFlow(packets=pkts, protocol="tcp",
                      flow_key=("10.0.0.1", "10.0.0.2", 12345, 80, "tcp"), pcap_path=None)
    assert SemanticChecker().check_flow(flow) is True


def test_flow_mixed_protocols_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    pkt1 = _make_tcp_packet(protocol="tcp")
    pkt2 = ParsedPacket(
        raw_packet=None, protocol="udp", ip_ttl=64, ip_tos=0, ip_total_length=28, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=20, icmp_type=None, icmp_code=None,
        timestamp=2000.0, payload_size=12,
    )
    flow = ParsedFlow(packets=[pkt1, pkt2], protocol="tcp",
                      flow_key=("10.0.0.1", "10.0.0.2", 12345, 80, "tcp"), pcap_path=None)
    assert SemanticChecker().check_flow(flow) is False


def test_flow_out_of_order_timestamps_fails():
    from pa_xai.pcap.semantic_checker import SemanticChecker
    pkts = [_make_tcp_packet(timestamp=2.0), _make_tcp_packet(timestamp=1.0)]
    flow = ParsedFlow(packets=pkts, protocol="tcp",
                      flow_key=("10.0.0.1", "10.0.0.2", 12345, 80, "tcp"), pcap_path=None)
    assert SemanticChecker().check_flow(flow) is False
