"""Async UDP listener for receiving raw sensor data.

Supports multiple packet formats (CSV, JSON, binary, raw bytes) and
auto-detection. Runs in an asyncio event loop in a background thread.
"""

import asyncio
import json
import struct
import threading
import time
import logging
from typing import Optional, Callable

import numpy as np

from autolabeler.acquisition.ring_buffer import RingBuffer
from autolabeler.config import HardwareConfig


logger = logging.getLogger(__name__)


class _UDPProtocol(asyncio.DatagramProtocol):
    """Internal asyncio protocol for receiving UDP datagrams."""

    def __init__(self, callback: Callable[[bytes], None]):
        self._callback = callback

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        self._callback(data)

    def error_received(self, exc: Exception) -> None:
        logger.error(f"UDP error: {exc}")


class UDPListener:
    """Hardware-agnostic UDP listener for real-time sensor data.

    Binds to a UDP port, receives packets, parses them according to the
    hardware profile's packet format, and writes samples into a ring buffer.

    Args:
        ring_buffer: Target ring buffer for parsed samples.
        hardware_config: Hardware configuration specifying port, format, etc.
    """

    def __init__(self, ring_buffer: RingBuffer, hardware_config: HardwareConfig):
        self._buffer = ring_buffer
        self._config = hardware_config

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._running = False

        self._packets_received = 0
        self._parse_errors = 0

        # Select parser based on format
        self._parser = self._get_parser(hardware_config.packet_format)

    def _get_parser(self, fmt: str) -> Callable[[bytes], Optional[np.ndarray]]:
        """Return the appropriate packet parser function."""
        parsers = {
            "csv": self._parse_csv,
            "json": self._parse_json,
            "binary": self._parse_binary,
            "raw": self._parse_raw,
        }
        if fmt not in parsers:
            raise ValueError(
                f"Unknown packet format '{fmt}'. Supported: {list(parsers.keys())}"
            )
        return parsers[fmt]

    def _parse_csv(self, data: bytes) -> Optional[np.ndarray]:
        """Parse CSV-formatted packet: 'val1,val2,...\\n' per sample."""
        try:
            text = data.decode("utf-8").strip()
            if not text:
                return None

            lines = text.split("\n")
            samples = []
            sep = self._config.packet_separator
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                values = [float(v) for v in line.split(sep) if v.strip()]
                samples.append(values)

            if not samples:
                return None
            return np.array(samples, dtype=np.float64)
        except Exception:
            self._parse_errors += 1
            return None

    def _parse_json(self, data: bytes) -> Optional[np.ndarray]:
        """Parse JSON packet: {"samples": [[v1, v2], ...]} or [v1, v2, ...]."""
        try:
            obj = json.loads(data.decode("utf-8"))
            if isinstance(obj, dict):
                samples = obj.get("samples", obj.get("data", []))
            elif isinstance(obj, list):
                samples = obj
            else:
                return None

            if not samples:
                return None

            arr = np.array(samples, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            return arr
        except Exception:
            self._parse_errors += 1
            return None

    def _parse_binary(self, data: bytes) -> Optional[np.ndarray]:
        """Parse binary packet: packed numeric values."""
        try:
            dtype_map = {
                "int16": np.int16,
                "int32": np.int32,
                "float32": np.float32,
                "float64": np.float64,
                "uint16": np.uint16,
            }
            dt = dtype_map.get(self._config.sample_dtype, np.int16)
            byte_order = "<" if self._config.byte_order == "little" else ">"
            ndt = np.dtype(dt).newbyteorder(byte_order)

            arr = np.frombuffer(data, dtype=ndt).astype(np.float64)
            channels = self._config.channels
            if len(arr) % channels != 0:
                # Truncate to fit
                arr = arr[: len(arr) - len(arr) % channels]
            if len(arr) == 0:
                return None
            return arr.reshape(-1, channels)
        except Exception:
            self._parse_errors += 1
            return None

    def _parse_raw(self, data: bytes) -> Optional[np.ndarray]:
        """Parse raw bytes — interpret as unsigned 8-bit values."""
        try:
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float64)
            channels = self._config.channels
            if len(arr) % channels != 0:
                arr = arr[: len(arr) - len(arr) % channels]
            if len(arr) == 0:
                return None
            return arr.reshape(-1, channels)
        except Exception:
            self._parse_errors += 1
            return None

    def _on_packet(self, data: bytes) -> None:
        """Handle incoming UDP packet."""
        self._packets_received += 1
        samples = self._parser(data)
        if samples is not None:
            n = samples.shape[0]
            now = time.time()
            dt = n / self._config.sample_rate
            timestamps = np.linspace(now - dt, now, n, endpoint=False)
            self._buffer.write(samples, timestamps)

    async def _start_server(self) -> None:
        """Start the UDP endpoint (runs in asyncio loop)."""
        loop = asyncio.get_running_loop()
        self._transport, _ = await loop.create_datagram_endpoint(
            lambda: _UDPProtocol(self._on_packet),
            local_addr=(self._config.udp_host, self._config.udp_port),
        )
        logger.info(
            f"UDP listener started on {self._config.udp_host}:{self._config.udp_port} "
            f"(format={self._config.packet_format})"
        )

    def _run_loop(self) -> None:
        """Run the asyncio event loop in a background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._loop.run_forever()

    def start(self) -> None:
        """Start listening for UDP packets in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        # Give it a moment to bind
        time.sleep(0.1)

    def stop(self) -> None:
        """Stop the listener and clean up."""
        if not self._running:
            return
        self._running = False
        if self._transport:
            self._loop.call_soon_threadsafe(self._transport.close)
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info(
            f"UDP listener stopped. Packets received: {self._packets_received}, "
            f"parse errors: {self._parse_errors}"
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def packets_received(self) -> int:
        return self._packets_received

    @property
    def parse_errors(self) -> int:
        return self._parse_errors
