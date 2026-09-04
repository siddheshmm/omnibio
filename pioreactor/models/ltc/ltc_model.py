"""Liquid Time-Constant (LTC) sequence model for yeast bioreactor digital twin.

Combines:
  1. Continuous fluid mass balances (volume dynamics, dilution rate)
  2. Multi-channel input encoder (chemical pumps, temperature, UV)
  3. Continuous-time LTC recurrent core with liquid time constants
  4. Pure state decoder for 14 physical sensor channels (no direct control feedthrough)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ltc_cell import LTCCell


class LTCBioreactorTwin(nn.Module):
    """Continuous-time Liquid Time-Constant (LTC) model for Pioreactor yeast culture."""

    def __init__(
        self,
        input_dim: int = 6,         # 4 fluid inputs + temp + uv
        hidden_dim: int = 32,        # LTC state dimension
        num_sensors: int = 14,       # 14 physical sensor channels
        unfolding_steps: int = 2,    # Solver sub-steps per 5-min bin
        dt_min: float = 5.0,         # 5 minutes per macro observation step
        tau_min: float = 1.0,        # Minimum time constant (mins)
        tau_max: float = 60.0,       # Maximum time constant (mins)
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_sensors = num_sensors
        self.dt_min = dt_min

        # Initial state encoder from initial optical density & volume
        self.init_state_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
        )

        # LTC cell receives: [u(t) (input_dim), volume(1), dilution_rate(1)] -> input_dim + 2
        ltc_in_features = input_dim + 2
        self.ltc_cell = LTCCell(
            input_dim=ltc_in_features,
            hidden_dim=hidden_dim,
            unfolding_steps=unfolding_steps,
            dt=dt_min,
            tau_min=tau_min,
            tau_max=tau_max,
        )

        # Pure physical state decoder: decodes [x_LTC(t), volume(t)] -> 14 sensors
        # Strictly no direct feedthrough from u(t) to ensure physical causality
        decoder_in_features = hidden_dim + 1
        self.sensor_decoder = nn.Sequential(
            nn.Linear(decoder_in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_sensors),
        )

    def forward(
        self,
        u_seq: torch.Tensor,
        init_od: torch.Tensor,
        init_vol: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward rollout over sequence of length T.
        
        Args:
            u_seq: (batch_size, T, input_dim) normalized input trajectory
                   columns: [add_media, add_alt, remove_waste, dose_total, temp_c, uv_intensity]
            init_od: (batch_size, 1) initial normalized OD
            init_vol: (batch_size, 1) initial volume in mL
            
        Returns:
            dict containing:
              - 'y_pred': (batch_size, T, 14) predicted sensor trajectories (normalized space)
              - 'tau_sys': (batch_size, T, hidden_dim) instantaneous liquid time-constants
              - 'hidden': (batch_size, T, hidden_dim) hidden states
              - 'volume': (batch_size, T, 1) simulated continuous volume
        """
        batch_size, seq_len, _ = u_seq.shape
        device = u_seq.device

        # 1. Initialize hidden state from initial culture conditions
        init_features = torch.cat([init_od, init_vol / 15.0], dim=-1)
        x = self.init_state_net(init_features)

        # Continuous volume tracking
        vol = init_vol.clone()

        y_preds = []
        taus = []
        hiddens = []
        volumes = []

        for t in range(seq_len):
            u_t = u_seq[:, t, :]  # (batch_size, input_dim)

            # Fluid flows (indices 0, 1, 2)
            inflow = u_t[:, 0:1] + u_t[:, 1:2]
            outflow = u_t[:, 2:3]

            # Continuous mass balance for volume: V(t+dt) = V(t) + net_flow
            vol = torch.clamp(vol + (inflow - outflow), min=1.0, max=25.0)

            # Dilution rate: D = F_in / (V * dt)
            dilution = inflow / (vol * self.dt_min)

            # Features fed into LTC cell
            ltc_input = torch.cat([u_t, vol / 15.0, dilution], dim=-1)

            # Integrate LTC ODE across macro step dt_min
            x, tau_sys = self.ltc_cell(x, ltc_input)

            # Decode physical and biological state into 14 sensor dimensions
            state_features = torch.cat([x, vol / 15.0], dim=-1)
            y_pred = self.sensor_decoder(state_features)

            y_preds.append(y_pred)
            taus.append(tau_sys)
            hiddens.append(x)
            volumes.append(vol)

        return {
            "y_pred": torch.stack(y_preds, dim=1),
            "tau_sys": torch.stack(taus, dim=1),
            "hidden": torch.stack(hiddens, dim=1),
            "volume": torch.stack(volumes, dim=1),
        }
