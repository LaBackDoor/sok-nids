import numpy as np
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


def _make_flow():
    pkts = [
        _make_tcp_packet(timestamp=1.0, tcp_seq=1000, tcp_flags=0x02),
        _make_tcp_packet(timestamp=2.0, tcp_seq=2000, tcp_flags=0x10),
    ]
    return ParsedFlow(
        packets=pkts, protocol="tcp",
        flow_key=("10.0.0.1", "10.0.0.2", 12345, 80, "tcp"),
        pcap_path=None,
    )


def test_flow_perturbator_returns_correct_count():
    from pa_xai.pcap.flow_perturbation import FlowPerturbator
    from pa_xai.pcap.perturbation import PacketPerturbator
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer, FlowConstraintEnforcer

    flow = _make_flow()
    fp = FlowPerturbator(PacketPerturbator(), FlowConstraintEnforcer(PacketConstraintEnforcer()))
    results = fp.perturb(flow, sigma=5.0, num_samples=10)
    assert len(results) == 10


def test_flow_perturbator_preserves_protocol():
    from pa_xai.pcap.flow_perturbation import FlowPerturbator
    from pa_xai.pcap.perturbation import PacketPerturbator
    from pa_xai.pcap.packet_constraints import PacketConstraintEnforcer, FlowConstraintEnforcer

    flow = _make_flow()
    fp = FlowPerturbator(PacketPerturbator(), FlowConstraintEnforcer(PacketConstraintEnforcer()))
    results = fp.perturb(flow, sigma=5.0, num_samples=10)
    for f in results:
        assert f.protocol == "tcp"
        for pkt in f.packets:
            assert pkt.protocol == "tcp"


def test_pipeline_generate_packet_neighborhood():
    from pa_xai.pcap.pipeline import PcapPipeline

    pkt = _make_tcp_packet()
    pipeline = PcapPipeline()
    results = pipeline.generate_neighborhood_from_packets([pkt], num_samples=50, sigma=5.0)
    assert len(results) == 50
    for r in results:
        assert isinstance(r, ParsedPacket)


def test_pipeline_generate_flow_neighborhood():
    from pa_xai.pcap.pipeline import PcapPipeline

    flow = _make_flow()
    pipeline = PcapPipeline()
    results = pipeline.generate_neighborhood_from_flow(flow, num_samples=10, sigma=5.0)
    assert len(results) == 10
    for f in results:
        assert isinstance(f, ParsedFlow)
        assert f.protocol == "tcp"
