"""Hardware profile management — YAML-based configs for known and generic devices."""

from pathlib import Path
from typing import Optional
import logging

from autolabeler.config import HardwareConfig, HARDWARE_PROFILES_DIR


logger = logging.getLogger(__name__)


# --- Built-in profiles ---
BUILTIN_PROFILES: dict[str, HardwareConfig] = {
    "plant_spikerbox": HardwareConfig(
        name="Plant SpikerBox",
        sample_rate=10000,
        channels=1,
        udp_port=5000,
        packet_format="csv",
        packet_separator=",",
        description="Backyard Brains Plant SpikerBox for plant biosignal recording",
    ),
    "neuron_spikerbox": HardwareConfig(
        name="Neuron SpikerBox",
        sample_rate=10000,
        channels=1,
        udp_port=5001,
        packet_format="csv",
        packet_separator=",",
        description="Backyard Brains Neuron SpikerBox for neural signal recording",
    ),
    "microphone": HardwareConfig(
        name="Microphone",
        sample_rate=44100,
        channels=1,
        udp_port=5002,
        packet_format="csv",
        packet_separator=",",
        description="Audio microphone — use tools/audio_bridge.py to forward mic input",
    ),
    "generic": HardwareConfig(
        name="Generic Sensor",
        sample_rate=1000,
        channels=1,
        udp_port=5000,
        packet_format="csv",
        packet_separator=",",
        description="Generic sensor — configure sample rate, channels, and format manually",
    ),
}


def list_profiles() -> dict[str, HardwareConfig]:
    """List all available hardware profiles (built-in + user YAML files).

    Returns:
        Dictionary of profile_name -> HardwareConfig.
    """
    profiles = dict(BUILTIN_PROFILES)

    # Load user-defined YAML profiles
    if HARDWARE_PROFILES_DIR.exists():
        for yaml_path in HARDWARE_PROFILES_DIR.glob("*.yaml"):
            try:
                config = HardwareConfig.from_yaml(yaml_path)
                name = yaml_path.stem
                profiles[name] = config
                logger.debug(f"Loaded hardware profile: {name} from {yaml_path}")
            except Exception as e:
                logger.warning(f"Failed to load hardware profile {yaml_path}: {e}")

    return profiles


def get_profile(name: str) -> Optional[HardwareConfig]:
    """Get a hardware profile by name."""
    profiles = list_profiles()
    return profiles.get(name)


def save_profile(name: str, config: HardwareConfig) -> Path:
    """Save a hardware profile as a YAML file."""
    HARDWARE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = HARDWARE_PROFILES_DIR / f"{name}.yaml"
    config.to_yaml(path)
    logger.info(f"Saved hardware profile: {name} to {path}")
    return path


def ensure_builtin_profiles() -> None:
    """Write built-in profiles to disk if they don't exist yet."""
    HARDWARE_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    for name, config in BUILTIN_PROFILES.items():
        path = HARDWARE_PROFILES_DIR / f"{name}.yaml"
        if not path.exists():
            config.to_yaml(path)
            logger.debug(f"Created built-in profile: {path}")
