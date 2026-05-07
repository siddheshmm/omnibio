"""Fake UDP data sender for testing AutoLabeler without real hardware.

Generates synthetic sinusoidal signals and sends them as CSV-formatted
UDP packets to simulate a sensor device.

Usage:
    python tools/fake_sender.py [--port 5000] [--rate 1000] [--channels 1]
"""

import argparse
import math
import socket
import time
import sys


def main():
    parser = argparse.ArgumentParser(description="Fake UDP sensor data sender")
    parser.add_argument("--host", default="127.0.0.1", help="UDP target host")
    parser.add_argument("--port", type=int, default=5000, help="UDP target port")
    parser.add_argument("--rate", type=int, default=1000, help="Sample rate (Hz)")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels")
    parser.add_argument("--freq", type=float, default=2.0, help="Signal frequency (Hz)")
    parser.add_argument("--amplitude", type=float, default=500.0, help="Signal amplitude")
    parser.add_argument("--noise", type=float, default=50.0, help="Noise amplitude")
    parser.add_argument("--batch", type=int, default=50, help="Samples per UDP packet")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.host, args.port)

    print(f"Sending fake data to {args.host}:{args.port}")
    print(f"  Rate: {args.rate} Hz, Channels: {args.channels}")
    print(f"  Signal: {args.freq} Hz sine, amplitude={args.amplitude}, noise={args.noise}")
    print(f"  Batch size: {args.batch} samples/packet")
    print("Press Ctrl+C to stop.\n")

    sample_idx = 0
    batch_interval = args.batch / args.rate

    import random

    try:
        while True:
            lines = []
            for i in range(args.batch):
                t = (sample_idx + i) / args.rate
                values = []
                for ch in range(args.channels):
                    # Each channel gets a slightly different frequency
                    freq = args.freq * (1 + ch * 0.3)
                    val = args.amplitude * math.sin(2 * math.pi * freq * t)
                    val += random.gauss(0, args.noise)
                    values.append(f"{val:.2f}")
                lines.append(",".join(values))

            packet = "\n".join(lines).encode("utf-8")
            sock.sendto(packet, addr)
            sample_idx += args.batch

            elapsed = sample_idx / args.rate
            if sample_idx % (args.rate * 5) == 0:
                print(f"  Sent {sample_idx} samples ({elapsed:.1f}s)")

            time.sleep(batch_interval)

    except KeyboardInterrupt:
        print(f"\nStopped after {sample_idx} samples ({sample_idx / args.rate:.1f}s)")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
