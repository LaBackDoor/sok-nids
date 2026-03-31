from pa_xai.pcap.parser import ParsedPacket


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


def test_pin_protocol():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = _make_tcp_packet()
    perturbed = _make_tcp_packet(protocol="udp")
    enforcer = PacketConstraintEnforcer()
    result = enforcer.enforce(perturbed, original)
    assert result.protocol == "tcp"


def test_clamp_ttl_to_valid_range():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = _make_tcp_packet(ip_ttl=64)
    enforcer = PacketConstraintEnforcer()

    perturbed = _make_tcp_packet(ip_ttl=300)
    result = enforcer.enforce(perturbed, original)
    assert 1 <= result.ip_ttl <= 255

    perturbed2 = _make_tcp_packet(ip_ttl=-5)
    result2 = enforcer.enforce(perturbed2, original)
    assert result2.ip_ttl >= 1


def test_tcp_flags_syn_repair():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = _make_tcp_packet(tcp_flags=0x02)
    perturbed = _make_tcp_packet(tcp_flags=0x02 | 0x01 | 0x04)
    enforcer = PacketConstraintEnforcer()
    result = enforcer.enforce(perturbed, original)
    assert result.tcp_flags & 0x02  # SYN set
    assert not (result.tcp_flags & 0x01)  # FIN off
    assert not (result.tcp_flags & 0x04)  # RST off


def test_urgent_ptr_zeroed_when_urg_off():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = _make_tcp_packet(tcp_flags=0x10)
    perturbed = _make_tcp_packet(tcp_flags=0x10, tcp_urgent_ptr=100)
    enforcer = PacketConstraintEnforcer()
    result = enforcer.enforce(perturbed, original)
    assert result.tcp_urgent_ptr == 0


def test_ip_total_length_recomputed():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = _make_tcp_packet(payload_size=20)
    perturbed = _make_tcp_packet(ip_total_length=9999, payload_size=20)
    enforcer = PacketConstraintEnforcer()
    result = enforcer.enforce(perturbed, original)
    assert result.ip_total_length == 60  # 20+20+20


def test_protocol_gating_tcp_fields_none_for_udp():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = ParsedPacket(
        raw_packet=None, protocol="udp",
        ip_ttl=64, ip_tos=0, ip_total_length=40, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=32, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=24,
    )
    perturbed = ParsedPacket(
        raw_packet=None, protocol="udp",
        ip_ttl=64, ip_tos=0, ip_total_length=40, ip_flags=0,
        tcp_window_size=65535, tcp_flags=0x10, tcp_seq=1000, tcp_ack=500,
        tcp_urgent_ptr=0, udp_length=32, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=24,
    )
    enforcer = PacketConstraintEnforcer()
    result = enforcer.enforce(perturbed, original)
    assert result.tcp_window_size is None
    assert result.tcp_flags is None


def test_icmp_type_snapped_to_valid():
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer
    original = ParsedPacket(
        raw_packet=None, protocol="icmp",
        ip_ttl=64, ip_tos=0, ip_total_length=28, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=None, icmp_type=8, icmp_code=0,
        timestamp=1000.0, payload_size=0,
    )
    perturbed = ParsedPacket(
        raw_packet=None, protocol="icmp",
        ip_ttl=64, ip_tos=0, ip_total_length=28, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=None, icmp_type=200, icmp_code=150,
        timestamp=1000.0, payload_size=0,
    )
    enforcer = PacketConstraintEnforcer()
    result = enforcer.enforce(perturbed, original)
    assert result.icmp_type in {0, 3, 5, 8, 11, 12}
