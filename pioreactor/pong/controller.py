import json
import os
import pickle
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None


MQTT_HOST = os.environ.get("PIOREACTOR_MQTT_HOST", "localhost")
MQTT_PORT = int(os.environ.get("PIOREACTOR_MQTT_PORT", "1883"))
UNIT = os.environ.get("PIOREACTOR_UNIT", "pioreactor01")
EXPERIMENT = os.environ.get("PIOREACTOR_EXPERIMENT", "demo")
TOPIC_BASE = f"pioreactor/{UNIT}/{EXPERIMENT}"
PIO_HOST = os.environ.get("PONG_PIO_HOST")
PIO_USER = os.environ.get("PONG_PIO_USER", "pioreactor")
PIO_BIN = os.environ.get("PONG_PIO_BIN", "pio")

MODEL_PATH = Path(__file__).with_name("ridge_model.pkl")
PUMP_COMMANDS = {
    "media": os.environ.get("PONG_MEDIA_COMMAND", "add_media"),
    "salt": os.environ.get("PONG_SALT_COMMAND", "add_alt_media"),
    "waste": os.environ.get("PONG_WASTE_COMMAND", "remove_waste"),
}
SENSOR_JOBS = [
    job.strip()
    for job in os.environ.get(
        "PONG_SENSOR_JOBS",
        "stirring,od_reading,growth_rate_calculating,spectrometer_reading",
    ).split(",")
    if job.strip()
]
WASTE_MULTIPLIER = float(os.environ.get("PONG_WASTE_MULTIPLIER", "1.0"))
DRY_RUN = os.environ.get("PONG_DRY_RUN", "0") == "1"
DEBUG = os.environ.get("PONG_DEBUG", "0") == "1"
MQTT_DEBUG = os.environ.get("PONG_MQTT_DEBUG", "0") == "1"


_sensor_data: Dict[str, float] = {}
_sensor_times: Dict[str, float] = {}
_topic_counts: Dict[str, int] = {}
_sensor_lock = threading.Lock()
_readout = None


def load_readout(model_path: Path = MODEL_PATH):
    with model_path.open("rb") as f:
        data = pickle.load(f)

    required = {"model", "scaler", "sensor_names"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{model_path} is missing keys: {sorted(missing)}")

    return data["model"], data["scaler"], list(data["sensor_names"])


def _coerce_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _store_feature(name: str, value) -> None:
    value = _coerce_float(value)
    if value is None:
        return
    with _sensor_lock:
        _sensor_data[name] = value
        _sensor_times[name] = time.time()


def _parse_sensor_payload(topic: str, payload: dict) -> None:
    topic_leaf = topic.split("/")[-1].lower()

    if "od_filtered" in topic_leaf or "normalized_od" in topic_leaf:
        value = payload.get("normalized_od_reading")
        if value is None:
            value = payload.get("norm_od")
        _store_feature("norm_od", value)
        return

    if "growth" in topic_leaf:
        value = payload.get("rate")
        if value is None:
            value = payload.get("growth_rate")
        _store_feature("growth_rate", value)
        return

    if "spectrum" in topic_leaf or "as7341" in topic_leaf:
        band = payload.get("band") or payload.get("channel")
        reading = payload.get("reading")
        if reading is None:
            reading = payload.get("intensity")
        if band is not None:
            _store_feature(f"nm_{int(float(band))}", reading)
        return

    if "od" in topic_leaf:
        angle = payload.get("angle")
        reading = payload.get("od_reading")
        if reading is None:
            reading = payload.get("reading")
        if angle is not None:
            _store_feature(f"OD_{int(float(angle))}", reading)


def on_message(client, userdata, msg):
    try:
        with _sensor_lock:
            _topic_counts[msg.topic] = _topic_counts.get(msg.topic, 0) + 1
        if MQTT_DEBUG:
            print(f"MQTT {msg.topic}: {msg.payload.decode(errors='replace')[:180]}")
        payload = json.loads(msg.payload.decode())
        if not isinstance(payload, dict):
            return
        _parse_sensor_payload(msg.topic, payload)
    except Exception as exc:
        print(f"MQTT parse error on {msg.topic}: {exc}")


def missing_features(feature_names: Iterable[str]) -> list:
    with _sensor_lock:
        return [name for name in feature_names if name not in _sensor_data]


def build_X(feature_names: Iterable[str]) -> Optional[np.ndarray]:
    feature_names = list(feature_names)
    with _sensor_lock:
        if any(name not in _sensor_data for name in feature_names):
            return None
        values = [_sensor_data[name] for name in feature_names]
    return np.array(values, dtype=float)


def sensor_summary(feature_names: Iterable[str]) -> dict:
    feature_names = list(feature_names)
    now = time.time()
    with _sensor_lock:
        present = [name for name in feature_names if name in _sensor_data]
        missing = [name for name in feature_names if name not in _sensor_data]
        ages = {
            name: round(now - _sensor_times[name], 1)
            for name in present
            if name in _sensor_times
        }
        recent_topics = sorted(
            _topic_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
    return {
        "present": present,
        "missing": missing,
        "ages": ages,
        "recent_topics": recent_topics,
    }


def predict_readout() -> Tuple[Optional[float], list]:
    global _readout
    if _readout is None:
        _readout = load_readout()

    model, scaler, sensor_names = _readout
    X = build_X(sensor_names)
    if X is None:
        return None, missing_features(sensor_names)

    Xs = scaler.transform(X.reshape(1, -1))
    y = float(model.predict(Xs)[0])
    return max(0.0, min(1.0, y)), []


def _run_pio(command: str, volume_ml: float) -> None:
    volume_ml = max(0.0, float(volume_ml))
    if volume_ml <= 0.0:
        return

    pio_cmd = [PIO_BIN, "run", command, "--ml", f"{volume_ml:.3f}"]
    cmd = pio_cmd
    if PIO_HOST:
        remote_script = shlex.join(pio_cmd)
        cmd = ["ssh", f"{PIO_USER}@{PIO_HOST}", f"bash -lc {shlex.quote(remote_script)}"]

    if DRY_RUN:
        print("DRY RUN:", " ".join(cmd))
        return

    if PIO_HOST:
        if shutil.which("ssh") is None:
            raise RuntimeError("Could not find ssh. Install OpenSSH or run on the Pioreactor itself.")
    elif shutil.which("pio") is None:
        raise RuntimeError("Could not find the Pioreactor 'pio' command. Set PONG_DRY_RUN=1 to test without pumps.")

    subprocess.run(cmd, check=True)


def _run_background_job(job_name: str) -> None:
    log_path = f"/tmp/pong_{job_name}.log"
    remote_script = f"nohup pio run {job_name} > {log_path} 2>&1 &"

    if PIO_HOST:
        cmd = ["ssh", f"{PIO_USER}@{PIO_HOST}", f"bash -lc {shlex.quote(remote_script)}"]
    else:
        cmd = ["bash", "-lc", remote_script]

    if DRY_RUN:
        print("DRY RUN:", " ".join(cmd))
        return

    if PIO_HOST and shutil.which("ssh") is None:
        raise RuntimeError("Could not find ssh. Install OpenSSH or run on the Pioreactor itself.")

    subprocess.run(cmd, check=True, timeout=20)


def start_sensor_jobs(jobs: Optional[Iterable[str]] = None) -> None:
    jobs = list(jobs or SENSOR_JOBS)
    for job in jobs:
        _run_background_job(job)


def dose_once(pump: str, volume_ml: float, remove_waste: bool = True) -> None:
    pump = pump.lower()
    if pump not in PUMP_COMMANDS:
        raise ValueError(f"Unknown pump '{pump}'. Expected one of {sorted(PUMP_COMMANDS)}")

    _run_pio(PUMP_COMMANDS[pump], volume_ml)

    if remove_waste and pump in {"media", "salt"} and WASTE_MULTIPLIER > 0:
        _run_pio(PUMP_COMMANDS["waste"], volume_ml * WASTE_MULTIPLIER)


def _set_status(game_state, status: str) -> None:
    setattr(game_state, "controller_status", status)


def run_controller(game_state, poll_seconds: float = 5.0):
    try:
        _, _, sensor_names = load_readout()
        _set_status(game_state, f"readout loaded: {len(sensor_names)} features")
    except Exception as exc:
        _set_status(game_state, f"readout load failed: {exc}")
        print(f"Controller stopped: {exc}")
        return

    client = None
    if mqtt is None:
        _set_status(game_state, "paho-mqtt missing; no live sensor stream")
        print("Install paho-mqtt on the Pioreactor to receive live sensor readings.")
    else:
        try:
            client = mqtt.Client()
            client.on_message = on_message
            client.connect(MQTT_HOST, MQTT_PORT)
            client.subscribe(f"{TOPIC_BASE}/#")
            client.loop_start()
            _set_status(game_state, f"listening on {TOPIC_BASE}/#")
            print(f"Controller listening on mqtt://{MQTT_HOST}:{MQTT_PORT}/{TOPIC_BASE}/#")
        except Exception as exc:
            _set_status(game_state, f"MQTT connect failed: {exc}")
            print(f"MQTT connect failed: {exc}")

    last_print = 0.0
    while True:
        prediction, missing = predict_readout()
        if prediction is None:
            setattr(game_state, "missing_features", missing)
            setattr(game_state, "controller_status", f"waiting for sensors: {len(missing)} missing")
        else:
            game_state.paddle.set_bio_output(prediction)
            setattr(game_state, "last_prediction", prediction)
            setattr(game_state, "missing_features", [])
            setattr(game_state, "last_sensor_update", time.time())
            setattr(game_state, "controller_status", f"ridge={prediction:.3f}")
            game_state.waiting_bio = False

        if DEBUG and time.time() - last_print >= 30.0:
            summary = sensor_summary(sensor_names)
            if prediction is None:
                print(
                    f"READOUT waiting: present={len(summary['present'])}/{len(sensor_names)} "
                    f"missing={summary['missing']}"
                )
            else:
                print(f"READOUT ridge={prediction:.3f} present={len(summary['present'])}/{len(sensor_names)}")
            if summary["recent_topics"]:
                print("MQTT topics:", summary["recent_topics"])
            last_print = time.time()

        time.sleep(poll_seconds)
