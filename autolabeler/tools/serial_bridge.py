"""Bridge: Serial (COM port) → UDP forwarder.

Reads raw data from a serial/COM device and forwards it as CSV-formatted
UDP packets to AutoLabeler. Works with any serial device including:
  - Backyard Brains Plant SpikerBox (10-bit encoded frames)
  - Backyard Brains Neuron SpikerBox
  - Arduino boards with Serial.println() output
  - Any device that streams via USB-to-serial

Requirements:
    pip install pyserial

Usage:
    # Plant SpikerBox on COM9 (auto-detected protocol)
    python tools/serial_bridge.py --port COM9

    # Arduino sending line-delimited values on COM3
    python tools/serial_bridge.py --port COM3 --baud 9600 --protocol raw_lines

    # List available COM ports
    python tools/serial_bridge.py --list
"""

import argparse
import socket
import time
import sys
import struct

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial")
    sys.exit(1)


def list_ports():
    """Print available serial/COM ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    print("Available serial ports:")
    for p in sorted(ports, key=lambda x: x.device):
        print(f"  {p.device:10s} — {p.description} [{p.hwid}]")


def bridge_spikerbox(ser, sock, addr, rate, batch_size):
    """Read Plant SpikerBox 10-bit encoded frames and forward as CSV UDP.

    Protocol: 2 bytes per sample.
      - Byte 1: MSB set (>= 128), contains 3 MSBs of 10-bit sample
      - Byte 2: MSB clear (< 128), contains 7 LSBs
      - Sample = ((byte1 & 0x7F) << 7) | (byte2 & 0x7F)
    """
    print(f"Protocol: spikerbox_plant (10-bit, 2-byte frames)")

    batch = []
    has_high = False
    high_byte = 0
    sample_count = 0
    batch_interval = batch_size / rate
    last_send = time.time()

    while True:
        waiting = ser.in_waiting
        if waiting < 2:
            # Flush batch if we've waited long enough
            if batch and (time.time() - last_send) > batch_interval:
                packet = "\n".join(f"{v:.1f}" for v in batch).encode("utf-8")
                sock.sendto(packet, addr)
                sample_count += len(batch)
                batch = []
                last_send = time.time()
            time.sleep(0.001)
            continue

        data = ser.read(min(waiting, 4096))

        for byte_val in data:
            if byte_val >= 128:
                # High byte (start of frame)
                high_byte = byte_val
                has_high = True
            elif has_high:
                # Low byte → reconstruct sample
                sample = ((high_byte & 0x7F) << 7) | (byte_val & 0x7F)
                batch.append(float(sample))
                has_high = False

                if len(batch) >= batch_size:
                    packet = "\n".join(f"{v:.1f}" for v in batch).encode("utf-8")
                    sock.sendto(packet, addr)
                    sample_count += len(batch)
                    batch = []
                    last_send = time.time()

                    if sample_count % (rate * 5) == 0:
                        print(f"  Forwarded {sample_count} samples "
                              f"({sample_count / rate:.1f}s)")


def bridge_raw_lines(ser, sock, addr, rate, batch_size):
    """Read newline-delimited ASCII values and forward as CSV UDP.

    Works with Arduino Serial.println() and similar devices.
    Each line can be a single value or comma-separated multi-channel values.
    """
    print(f"Protocol: raw_lines (ASCII, newline-delimited)")

    batch = []
    sample_count = 0

    while True:
        line = ser.readline()
        if not line:
            continue

        text = line.decode("utf-8", errors="ignore").strip()
        if not text:
            continue

        batch.append(text)

        if len(batch) >= batch_size:
            packet = "\n".join(batch).encode("utf-8")
            sock.sendto(packet, addr)
            sample_count += len(batch)
            batch = []

            if sample_count % (rate * 5) == 0:
                print(f"  Forwarded {sample_count} samples "
                      f"({sample_count / rate:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Serial → UDP bridge for AutoLabeler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/serial_bridge.py --list\n"
            "  python tools/serial_bridge.py --port COM9\n"
            "  python tools/serial_bridge.py --port COM3 --baud 9600 --protocol raw_lines\n"
        ),
    )
    parser.add_argument("--list", action="store_true", help="List available COM ports and exit")
    parser.add_argument("--port", default="COM9", help="Serial/COM port (default: COM9)")
    parser.add_argument("--baud", type=int, default=222222, help="Baud rate (default: 222222)")
    parser.add_argument("--protocol", default="spikerbox_plant",
                        choices=["spikerbox_plant", "raw_lines"],
                        help="Serial data protocol (default: spikerbox_plant)")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP target host")
    parser.add_argument("--udp-port", type=int, default=5000, help="UDP target port")
    parser.add_argument("--rate", type=int, default=10000, help="Expected sample rate in Hz")
    parser.add_argument("--batch", type=int, default=200, help="Samples per UDP packet")
    args = parser.parse_args()

    if args.list:
        list_ports()
        return

    # Open serial port
    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(port=args.port, baudrate=args.baud, timeout=1.0)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {args.port}: {e}")
        print("\nAvailable ports:")
        list_ports()
        sys.exit(1)

    ser.reset_input_buffer()

    # Open UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.udp_host, args.udp_port)

    print(f"Forwarding {args.port} → UDP {args.udp_host}:{args.udp_port}")
    print(f"  Baud: {args.baud}, Rate: {args.rate} Hz, Batch: {args.batch}")
    print("Press Ctrl+C to stop.\n")

    try:
        if args.protocol == "spikerbox_plant":
            bridge_spikerbox(ser, sock, addr, args.rate, args.batch)
        elif args.protocol == "raw_lines":
            bridge_raw_lines(ser, sock, addr, args.rate, args.batch)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()
        sock.close()


if __name__ == "__main__":
    main()
