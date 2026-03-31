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


def _make_flow(packets, protocol="tcp"):
    return ParsedFlow(
        packets=packets, protocol=protocol,
        flow_key=("10.0.0.1", "10.0.0.2", 12345, 80, protocol),
        pcap_path=None,
    )


def test_flow_enforces_temporal_ordering():
    from pa_xai.pcap.packet_constraints import FlowConstraintEnforcer, PacketConstraintEnforcer
    pkts = [
        _make_tcp_packet(timestamp=3.0), _make_tcp_packet(timestamp=1.0), _make_tcp_packet(timestamp=2.0),
    ]
    orig_pkts = [_make_tcp_packet(timestamp=1.0), _make_tcp_packet(timestamp=2.0), _make_tcp_packet(timestamp=3.0)]
    flow = _make_flow(pkts)
    original = _make_flow(orig_pkts)
    enforcer = FlowConstraintEnforcer(PacketConstraintEnforcer())
    result = enforcer.enforce(flow, original)
    timestamps = [p.timestamp for p in result.packets]
    assert timestamps == sorted(timestamps)


def test_flow_enforces_protocol_homogeneity():
    from pa_xai.pcap.packet_constraints import FlowConstraintEnforcer, PacketConstraintEnforcer
    pkts = [_make_tcp_packet(protocol="tcp"), _make_tcp_packet(protocol="udp")]
    orig_pkts = [_make_tcp_packet(protocol="tcp"), _make_tcp_packet(protocol="tcp")]
    flow = _make_flow(pkts, protocol="tcp")
    original = _make_flow(orig_pkts, protocol="tcp")
    enforcer = FlowConstraintEnforcer(PacketConstraintEnforcer())
    result = enforcer.enforce(flow, original)
    for pkt in result.packets:
        assert pkt.protocol == "tcp"


def test_flow_repairs_tcp_sequences():
    from pa_xai.pcap.packet_constraints import FlowConstraintEnforcer, PacketConstraintEnforcer
    pkts = [
        _make_tcp_packet(timestamp=1.0, tcp_seq=9999, payload_size=100),
        _make_tcp_packet(timestamp=2.0, tcp_seq=5555, payload_size=50),
    ]
    orig_pkts = [
        _make_tcp_packet(timestamp=1.0, tcp_seq=1000, payload_size=100),
        _make_tcp_packet(timestamp=2.0, tcp_seq=1100, payload_size=50),
    ]
    flow = _make_flow(pkts)
    original = _make_flow(orig_pkts)
    enforcer = FlowConstraintEnforcer(PacketConstraintEnforcer())
    result = enforcer.enforce(flow, original)
    # After repair, seq[1] = seq[0] + max(payload[0], 1)
    assert result.packets[1].tcp_seq == result.packets[0].tcp_seq + 100
