import numpy as np
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


def test_perturbator_returns_correct_count():
    from pa_xai.pcap.perturbation import PacketPerturbator
    pkt = _make_tcp_packet()
    perturbator = PacketPerturbator()
    results = perturbator.perturb(pkt, sigma=5.0, num_samples=50)
    assert len(results) == 50


def test_perturbator_preserves_protocol():
    from pa_xai.pcap.perturbation import PacketPerturbator
    pkt = _make_tcp_packet()
    perturbator = PacketPerturbator()
    results = perturbator.perturb(pkt, sigma=5.0, num_samples=100)
    for r in results:
        assert r.protocol == "tcp"


def test_perturbator_varies_ttl():
    from pa_xai.pcap.perturbation import PacketPerturbator
    np.random.seed(42)
    pkt = _make_tcp_packet(ip_ttl=64)
    perturbator = PacketPerturbator()
    results = perturbator.perturb(pkt, sigma=10.0, num_samples=100)
    ttls = [r.ip_ttl for r in results]
    assert len(set(ttls)) > 1


def test_perturbator_does_not_perturb_ip_total_length():
    from pa_xai.pcap.perturbation import PacketPerturbator
    pkt = _make_tcp_packet(ip_total_length=60)
    perturbator = PacketPerturbator()
    results = perturbator.perturb(pkt, sigma=10.0, num_samples=50)
    for r in results:
        assert r.ip_total_length == 60


def test_perturbator_tcp_none_for_udp():
    from pa_xai.pcap.perturbation import PacketPerturbator
    pkt = ParsedPacket(
        raw_packet=None, protocol="udp",
        ip_ttl=64, ip_tos=0, ip_total_length=40, ip_flags=0,
        tcp_window_size=None, tcp_flags=None, tcp_seq=None, tcp_ack=None,
        tcp_urgent_ptr=None, udp_length=32, icmp_type=None, icmp_code=None,
        timestamp=1000.0, payload_size=24,
    )
    perturbator = PacketPerturbator()
    results = perturbator.perturb(pkt, sigma=5.0, num_samples=50)
    for r in results:
        assert r.tcp_flags is None
        assert r.tcp_seq is None
