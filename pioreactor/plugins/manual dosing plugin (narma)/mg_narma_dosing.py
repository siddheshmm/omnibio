# -*- coding: utf-8 -*-
"""
mg_narma_dosing.py  — v1.1.0
─────────────────────────────
Drives the media pump with a pre-generated time series — either
Mackey-Glass (chaotic) or NARMA-10 (nonlinear stochastic) — scaled to
[min_volume_ml, max_volume_ml] per cycle.  Equal waste is removed each
cycle to keep working volume constant.

Two SQLite tables are written:
  mg_narma_input_series  — full pre-generated reference series (written once at startup)
  mg_narma_volumes       — live per-cycle volumes with timestamps (for UI chart)

CLI usage:
    pio run dosing_automation \
        --automation-name mg_narma_dosing \
        --duration 5 \
        --series-type mackey_glass \
        --total-cycles 72 \
        --min-volume-ml 0.1 \
        --max-volume-ml 0.9 \
        --mg-tau 17 \
        --seed 42

    pio run dosing_automation \
        --automation-name mg_narma_dosing \
        --duration 10 \
        --series-type mackey_glass_periodic \
        --total-cycles 48 \
        --mg-period-cycles 24 \
        --min-volume-ml 0.05 \
        --max-volume-ml 0.35 \
        --mg-tau 17
"""
from __future__ import annotations
import time

from pioreactor.automations.dosing.base import DosingAutomationJobContrib
from pioreactor.automations import events
from pioreactor.actions.pump import add_media, remove_waste
from pioreactor.utils import sqlite_worker
from pioreactor.utils.timing import current_utc_timestamp
from pioreactor.config import config


__plugin_summary__ = "Mackey-Glass / NARMA-10 time-series encoding dosing"
__plugin_version__ = "1.2.0"
__plugin_name__    = "MG NARMA Dosing"
__plugin_author__  = "Siddhesh"
__plugin_homepage__ = "https://pioreactor.com"


# ── Custom events ─────────────────────────────────────────────────────────────

class TimeSeriesDoseEvent(events.AutomationEvent):
    pass

class AllCyclesCompleteEvent(events.AutomationEvent):
    pass


# ── Time series generators ────────────────────────────────────────────────────

def _generate_mackey_glass(n_steps, tau=17, beta=0.2, gamma=0.1, n=10.0, warmup=500):
    """
    Euler integration of the Mackey-Glass DDE:
        dx/dt = beta * x(t-tau) / (1 + x(t-tau)^n)  -  gamma * x(t)
    Initial condition: x = 1.2 for all t <= 0.
    Returns a list of length n_steps.
    """
    import numpy as np
    total = warmup + n_steps
    buf   = tau + 1
    x     = [1.2] * (buf + total)
    for t in range(buf, buf + total):
        xt    = x[t - tau]
        x[t]  = x[t - 1] + beta * xt / (1.0 + xt ** n) - gamma * x[t - 1]
    return np.array(x[buf + warmup:], dtype=float)


def _generate_narma10(n_steps, warmup=200, seed=42):
    """
    NARMA-10:  y(t+1) = 0.3*y(t) + 0.05*y(t)*sum(y[t-9:t]) + 1.5*u[t-9]*u[t] + 0.1
    u ~ Uniform(0, 0.5)
    Returns (u_array, y_array) each of length n_steps.
    """
    import numpy as np
    rng   = np.random.default_rng(seed)
    total = warmup + n_steps
    u     = rng.uniform(0.0, 0.5, size=total)
    y     = [0.0] * total
    for t in range(10, total):
        y[t] = (0.3 * y[t-1]
                + 0.05 * y[t-1] * sum(y[t-10:t-1])
                + 1.5 * u[t-9] * u[t]
                + 0.1)
    return np.array(u[warmup:], dtype=float), np.array(y[warmup:], dtype=float)


def _normalise(series, lo, hi):
    import numpy as np
    s0, s1 = series.min(), series.max()
    if abs(s1 - s0) < 1e-12:
        return np.full_like(series, (lo + hi) / 2.0)
    return lo + (series - s0) / (s1 - s0) * (hi - lo)


def _tile_to_length(series, n_steps):
    """
    Repeat one finite motif until n_steps values are available.
    Used for periodic Mackey-Glass input encoding.
    """
    import numpy as np
    if len(series) <= 0:
        raise ValueError("periodic series must contain at least one value")
    repeats = int(np.ceil(float(n_steps) / float(len(series))))
    return np.tile(series, repeats)[:n_steps]


# ── Dosing automation ─────────────────────────────────────────────────────────

class MGNARMADosing(DosingAutomationJobContrib):
    """
    Mackey-Glass / NARMA-10 time-series encoding dosing automation.

    Each cycle: remove waste → add media, both at volume = series[cycle].
    Stops automatically after total_cycles.
    """

    automation_name = "mg_narma_dosing"

    published_settings = {
        "series_type":      {"datatype": "string", "settable": False, "unit": None},
        "total_cycles":     {"datatype": "string", "settable": False, "unit": "cycles"},
        "min_volume_ml":    {"datatype": "float",  "settable": False, "unit": "mL"},
        "max_volume_ml":    {"datatype": "float",  "settable": False, "unit": "mL"},
        "mg_tau":           {"datatype": "string", "settable": False, "unit": None},
        "mg_period_cycles": {"datatype": "string", "settable": False, "unit": "cycles"},
        "seed":             {"datatype": "string", "settable": False, "unit": None},
        # Live reading shown in UI chart via mqtt_topic
        "current_volume_ml": {"datatype": "float", "settable": False, "unit": "mL"},
    }

    def __init__(
        self,
        series_type:   str   = "mackey_glass",
        total_cycles:  int   = 72,
        min_volume_ml: float = 0.1,
        max_volume_ml: float = 0.9,
        mg_tau:        int   = 17,
        mg_period_cycles: int = 24,
        mg_warmup:     int   = 500,
        seed:          int   = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.series_type    = str(series_type).lower().strip()
        self.total_cycles   = str(int(total_cycles))   # stored as str in published_settings
        self.min_volume_ml  = float(min_volume_ml)
        self.max_volume_ml  = float(max_volume_ml)
        self.mg_tau         = str(int(mg_tau))
        self.mg_period_cycles = str(int(mg_period_cycles))
        self.seed           = str(int(seed))
        self._total         = int(total_cycles)        # private int for comparisons
        self._cycle_count   = 0                        # private counter — not published
        self._narma_target  = None
        self.current_volume_ml = self.min_volume_ml

        if self._total <= 0:
            raise ValueError("total_cycles must be greater than 0")
        if self.min_volume_ml < 0 or self.max_volume_ml < 0:
            raise ValueError("min_volume_ml and max_volume_ml must be non-negative")
        if self.max_volume_ml < self.min_volume_ml:
            raise ValueError("max_volume_ml must be >= min_volume_ml")

        # ── generate series ───────────────────────────────────────────────
        if self.series_type == "mackey_glass":
            raw = _generate_mackey_glass(
                self._total, tau=int(mg_tau), warmup=int(mg_warmup)
            )
            self._volumes = _normalise(raw, self.min_volume_ml, self.max_volume_ml)

        elif self.series_type in ("mackey_glass_periodic", "periodic_mackey_glass"):
            period_cycles = int(mg_period_cycles)
            if period_cycles <= 1:
                raise ValueError("mg_period_cycles must be greater than 1")

            raw_period = _generate_mackey_glass(
                period_cycles, tau=int(mg_tau), warmup=int(mg_warmup)
            )
            volume_period = _normalise(
                raw_period, self.min_volume_ml, self.max_volume_ml
            )
            self._volumes = _tile_to_length(volume_period, self._total)

        elif self.series_type == "narma10":
            u_raw, y_raw = _generate_narma10(
                self._total, warmup=200, seed=int(seed)
            )
            self._volumes      = _normalise(u_raw, self.min_volume_ml, self.max_volume_ml)
            self._narma_target = y_raw.tolist()

        else:
            raise ValueError(
                "series_type must be 'mackey_glass', "
                "'mackey_glass_periodic', or 'narma10', "
                f"got '{self.series_type}'"
            )

        # ── set up SQLite tables ──────────────────────────────────────────
        db_path = config.get("storage", "database")
        self._db = sqlite_worker.Sqlite3Worker(db_path)

        # Reference table: full pre-generated series (written once at startup)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS mg_narma_input_series (
                experiment       TEXT NOT NULL,
                pioreactor_unit  TEXT NOT NULL,
                series_type      TEXT NOT NULL,
                cycle_index      INTEGER NOT NULL,
                volume_ml        REAL NOT NULL,
                narma_target_y   REAL,
                PRIMARY KEY (experiment, pioreactor_unit, cycle_index)
            )
        """)

        # Live table: one row per executed cycle with timestamp (used by UI chart)
        self._db.execute("""
            CREATE TABLE IF NOT EXISTS mg_narma_volumes (
                experiment       TEXT NOT NULL,
                pioreactor_unit  TEXT NOT NULL,
                timestamp        TEXT NOT NULL,
                volume_ml        REAL NOT NULL
            )
        """)

        self._save_reference_series()

        self.logger.info(
            "MG/NARMA Dosing v1.2.0 | series=%s | %d cycles | "
            "vol=[%.2f, %.2f] mL | tau=%s | period=%s | seed=%s"
            % (self.series_type, self._total,
               self.min_volume_ml, self.max_volume_ml,
               self.mg_tau, self.mg_period_cycles, self.seed)
        )

    # ── Persist reference series once at startup ──────────────────────────────

    def _save_reference_series(self):
        try:
            for i, vol in enumerate(self._volumes.tolist()):
                target_y = self._narma_target[i] if self._narma_target is not None else None
                self._db.execute(
                    "INSERT OR REPLACE INTO mg_narma_input_series "
                    "(experiment, pioreactor_unit, series_type, cycle_index, volume_ml, narma_target_y) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (self.experiment, self.unit, self.series_type, i, vol, target_y),
                )
            self.logger.info(
                "Saved %d reference values to mg_narma_input_series." % len(self._volumes)
            )
        except Exception as e:
            self.logger.warning("Could not save reference series to DB: %s" % e)

    # ── Persist live volume for UI chart ─────────────────────────────────────

    def _save_live_volume(self, volume_ml):
        try:
            self._db.execute(
                "INSERT INTO mg_narma_volumes "
                "(experiment, pioreactor_unit, timestamp, volume_ml) "
                "VALUES (?, ?, ?, ?)",
                (self.experiment, self.unit, current_utc_timestamp(), volume_ml),
            )
        except Exception as e:
            self.logger.debug("Could not save live volume to DB: %s" % e)

    # ── Pump kwargs ───────────────────────────────────────────────────────────

    @property
    def _pump_kwargs(self):
        return dict(
            unit=self.unit,
            experiment=self.experiment,
            source_of_event="%s:%s" % (self.job_name, self.automation_name),
            mqtt_client=self.pub_client,
            logger=self.logger,
        )

    # ── Execute (called every `duration` minutes) ─────────────────────────────

    def execute(self):
        if self._cycle_count >= self._total:
            self.logger.info(
                "All %d cycles complete. Stopping automation." % self._total
            )
            self.set_state(self.DISCONNECTED)
            return AllCyclesCompleteEvent(
                "Completed %d cycles." % self._total,
                {"total_cycles": self._total},
            )

        volume = float(self._volumes[self._cycle_count])
        self.current_volume_ml = volume
        self._save_live_volume(volume)

        self.logger.info(
            "Cycle %d/%d | %s | dosing %.4f mL"
            % (self._cycle_count + 1, self._total, self.series_type, volume)
        )

        # Waste first, then media (constant working volume)
        remove_waste(ml=volume, **self._pump_kwargs)
        add_media(ml=volume,    **self._pump_kwargs)

        self._cycle_count += 1

        return TimeSeriesDoseEvent(
            "Cycle %d/%d: %.4f mL" % (self._cycle_count, self._total, volume),
            {
                "volume_ml":   volume,
                "cycle":       self._cycle_count,
                "series_type": self.series_type,
            },
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def on_disconnected(self):
        try:
            self._db.close()
        except Exception:
            pass
