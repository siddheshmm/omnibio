"""Bridge: Audio (microphone) → UDP forwarder.

Captures audio from any system input device and forwards it as
CSV-formatted UDP packets to AutoLabeler. Works with:
  - USB microphones
  - Built-in laptop microphones
  - Audio interfaces (Focusrite, Behringer, etc.)
  - Any PortAudio-compatible input device

Requirements:
    pip install sounddevice

Usage:
    # Use system default microphone
    python tools/audio_bridge.py

    # List available audio devices
    python tools/audio_bridge.py --list

    # Use a specific device by index
    python tools/audio_bridge.py --device 3 --rate 44100

    # Forward to a specific UDP port
    python tools/audio_bridge.py --udp-port 5002
"""

import argparse
import socket
import sys
import time

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("ERROR: sounddevice is required. Install with: pip install sounddevice")
    sys.exit(1)


def list_devices():
    """Print available audio input devices."""
    devices = sd.query_devices()
    print("Available audio input devices:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " ★" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']} "
                  f"({dev['max_input_channels']}ch, "
                  f"{int(dev['default_samplerate'])}Hz){default}")


def main():
    parser = argparse.ArgumentParser(
        description="Audio → UDP bridge for AutoLabeler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/audio_bridge.py --list\n"
            "  python tools/audio_bridge.py\n"
            "  python tools/audio_bridge.py --device 3 --rate 44100\n"
        ),
    )
    parser.add_argument("--list", action="store_true", help="List audio input devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Audio device index (default: system default)")
    parser.add_argument("--rate", type=int, default=44100, help="Sample rate in Hz (default: 44100)")
    parser.add_argument("--channels", type=int, default=1, help="Number of channels (default: 1)")
    parser.add_argument("--udp-host", default="127.0.0.1", help="UDP target host")
    parser.add_argument("--udp-port", type=int, default=5000, help="UDP target port")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    # Open UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (args.udp_host, args.udp_port)

    sample_count = 0

    def audio_callback(indata, frames, time_info, status):
        """Called by sounddevice for each audio block."""
        nonlocal sample_count
        if status:
            print(f"  ⚠ Audio status: {status}", file=sys.stderr)

        # Convert to CSV lines
        lines = []
        for row in indata:
            values = ",".join(f"{v:.6f}" for v in row)
            lines.append(values)

        packet = "\n".join(lines).encode("utf-8")
        sock.sendto(packet, addr)
        sample_count += frames

    # Determine device info
    device = args.device
    if device is not None:
        info = sd.query_devices(device)
        dev_name = info["name"]
    else:
        info = sd.query_devices(sd.default.device[0])
        dev_name = f"{info['name']} (default)"

    blocksize = max(64, args.rate // 30)  # ~33ms blocks

    print(f"Audio bridge: {dev_name} → UDP {args.udp_host}:{args.udp_port}")
    print(f"  Rate: {args.rate} Hz, Channels: {args.channels}, Block: {blocksize}")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            device=device,
            samplerate=args.rate,
            channels=args.channels,
            dtype="float32",
            blocksize=blocksize,
            callback=audio_callback,
        ):
            while True:
                time.sleep(5.0)
                print(f"  Forwarded {sample_count} samples "
                      f"({sample_count / args.rate:.1f}s)")
    except KeyboardInterrupt:
        print(f"\nStopped after {sample_count} samples "
              f"({sample_count / args.rate:.1f}s)")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
