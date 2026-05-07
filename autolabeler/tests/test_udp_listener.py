"""Tests for the UDP listener module."""

import asyncio
import socket
import time
import numpy as np
import pytest

from autolabeler.acquisition.ring_buffer import RingBuffer
from autolabeler.acquisition.udp_listener import UDPListener
from autolabeler.config import HardwareConfig


def _send_udp(host: str, port: int, data: bytes) -> None:
    """Send a UDP packet."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, (host, port))
    sock.close()


class TestUDPListenerCSV:
    """Test CSV packet parsing via live UDP."""

    def test_receives_csv_packets(self):
        config = HardwareConfig(
            udp_port=15001,
            sample_rate=1000,
            channels=1,
            packet_format="csv",
        )
        buf = RingBuffer(max_samples=10000, channels=1)
        listener = UDPListener(buf, config)
        listener.start()

        try:
            time.sleep(0.2)  # Wait for bind
            # Send CSV data
            _send_udp("127.0.0.1", 15001, b"1.0\n2.0\n3.0\n")
            time.sleep(0.2)

            assert listener.packets_received >= 1
            assert buf.available_samples >= 3
            data, _ = buf.read_latest(3)
            np.testing.assert_array_equal(data.flatten(), [1.0, 2.0, 3.0])
        finally:
            listener.stop()

    def test_receives_multichannel_csv(self):
        config = HardwareConfig(
            udp_port=15002,
            sample_rate=1000,
            channels=2,
            packet_format="csv",
        )
        buf = RingBuffer(max_samples=10000, channels=2)
        listener = UDPListener(buf, config)
        listener.start()

        try:
            time.sleep(0.2)
            _send_udp("127.0.0.1", 15002, b"1.0,2.0\n3.0,4.0\n")
            time.sleep(0.2)

            assert buf.available_samples >= 2
            data, _ = buf.read_latest(2)
            np.testing.assert_array_equal(data, [[1.0, 2.0], [3.0, 4.0]])
        finally:
            listener.stop()


class TestUDPListenerJSON:
    """Test JSON packet parsing."""

    def test_receives_json_array(self):
        config = HardwareConfig(
            udp_port=15003,
            sample_rate=1000,
            channels=1,
            packet_format="json",
        )
        buf = RingBuffer(max_samples=10000, channels=1)
        listener = UDPListener(buf, config)
        listener.start()

        try:
            time.sleep(0.2)
            _send_udp("127.0.0.1", 15003, b'[5.0, 6.0, 7.0]')
            time.sleep(0.2)

            assert buf.available_samples >= 3
            data, _ = buf.read_latest(3)
            np.testing.assert_array_equal(data.flatten(), [5.0, 6.0, 7.0])
        finally:
            listener.stop()

    def test_receives_json_dict(self):
        config = HardwareConfig(
            udp_port=15004,
            sample_rate=1000,
            channels=1,
            packet_format="json",
        )
        buf = RingBuffer(max_samples=10000, channels=1)
        listener = UDPListener(buf, config)
        listener.start()

        try:
            time.sleep(0.2)
            _send_udp("127.0.0.1", 15004, b'{"samples": [10.0, 20.0]}')
            time.sleep(0.2)

            assert buf.available_samples >= 2
            data, _ = buf.read_latest(2)
            np.testing.assert_array_equal(data.flatten(), [10.0, 20.0])
        finally:
            listener.stop()


class TestUDPListenerBinary:
    """Test binary packet parsing."""

    def test_receives_binary_int16(self):
        config = HardwareConfig(
            udp_port=15005,
            sample_rate=1000,
            channels=1,
            packet_format="binary",
            sample_dtype="int16",
            byte_order="little",
        )
        buf = RingBuffer(max_samples=10000, channels=1)
        listener = UDPListener(buf, config)
        listener.start()

        try:
            time.sleep(0.2)
            # Send 3 int16 values in little-endian
            data_bytes = np.array([100, 200, 300], dtype="<i2").tobytes()
            _send_udp("127.0.0.1", 15005, data_bytes)
            time.sleep(0.2)

            assert buf.available_samples >= 3
            data, _ = buf.read_latest(3)
            np.testing.assert_array_equal(data.flatten(), [100.0, 200.0, 300.0])
        finally:
            listener.stop()


class TestUDPListenerLifecycle:
    """Test start/stop lifecycle."""

    def test_start_stop(self):
        config = HardwareConfig(udp_port=15006, packet_format="csv")
        buf = RingBuffer(max_samples=1000, channels=1)
        listener = UDPListener(buf, config)

        assert not listener.is_running
        listener.start()
        assert listener.is_running
        listener.stop()
        assert not listener.is_running

    def test_malformed_csv_increments_parse_errors(self):
        config = HardwareConfig(udp_port=15007, packet_format="csv", channels=2)
        buf = RingBuffer(max_samples=1000, channels=2)
        listener = UDPListener(buf, config)
        listener.start()

        try:
            time.sleep(0.2)
            # Send malformed data (wrong channel count)
            _send_udp("127.0.0.1", 15007, b"not_a_number\n")
            time.sleep(0.2)

            assert listener.parse_errors >= 1
        finally:
            listener.stop()
