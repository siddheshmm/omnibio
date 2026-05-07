"""Global configuration and defaults for AutoLabeler."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import yaml


# --- Default paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
HARDWARE_PROFILES_DIR = PROJECT_ROOT / "hardware_profiles"
CUSTOM_PIPELINES_DIR = PROJECT_ROOT / "custom_pipelines"


@dataclass
class HardwareConfig:
    """Configuration for a hardware device.

    Defines the UDP endpoint and packet format that the app listens on.
    Any device (serial, audio, custom sensor) sends its data to this
    UDP port via a bridge script.
    """
    name: str = "Generic Sensor"
    sample_rate: int = 1000
    channels: int = 1
    udp_port: int = 5000
    udp_host: str = "127.0.0.1"
    packet_format: str = "csv"        # csv, json, binary, raw
    packet_separator: str = ","
    byte_order: str = "little"        # for binary format
    sample_dtype: str = "int16"       # for binary format
    description: str = ""

    @classmethod
    def from_yaml(cls, path: Path) -> "HardwareConfig":
        """Load hardware config from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: Path) -> None:
        """Save hardware config to a YAML file."""
        from dataclasses import asdict
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment session."""
    subject_id: str = "subject_01"
    session_id: str = "session_01"
    classes: list[str] = field(default_factory=lambda: ["touch", "no_touch"])
    trial_duration: float = 5.0       # seconds per trial
    rest_duration: float = 3.0        # seconds of rest between trials
    trials_per_class: int = 30
    block_size: Optional[int] = None  # None = auto (= number of classes)
    randomize: bool = True
    countdown_duration: float = 5.0   # seconds before experiment starts

    @property
    def effective_block_size(self) -> int:
        """Block size defaults to number of classes if not specified."""
        return self.block_size or len(self.classes)

    @property
    def total_trials(self) -> int:
        return self.trials_per_class * len(self.classes)

    @property
    def estimated_duration(self) -> float:
        """Estimated total experiment duration in seconds."""
        return (self.total_trials * (self.trial_duration + self.rest_duration)
                + self.countdown_duration)


@dataclass
class MLConfig:
    """Configuration for the ML pipeline."""
    # Preprocessing
    dc_offset_removal: bool = True
    bandpass_filter: bool = False
    bandpass_low: float = 0.1
    bandpass_high: float = 500.0
    normalize: bool = False

    # Feature extraction
    features: list[str] = field(default_factory=lambda: [
        "rms", "peak_to_peak", "std", "p90", "mean_abs"
    ])

    # Window extraction
    window_duration: float = 5.0      # seconds
    window_margin: float = 0.5        # seconds of margin around event

    # Training
    models: list[str] = field(default_factory=lambda: [
        "random_forest", "logistic_regression", "gradient_boosting"
    ])
    cv_folds: int = 5                 # Stratified K-Fold
    random_state: int = 42


@dataclass
class AppConfig:
    """Top-level application configuration."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    # Ring buffer
    buffer_duration: float = 600.0    # seconds of signal to keep in memory

    # GUI
    plot_update_interval_ms: int = 33 # ~30 FPS for live signal
    plot_window_seconds: float = 5.0  # visible window duration
