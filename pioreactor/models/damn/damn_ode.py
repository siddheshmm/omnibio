"""Dynamic Artificial Metabolic Network (dAMN) ODE integration engine.

Pure continuous-time grey-box ODE integration combining CSTR fluid/mass balance
with a neural-mechanistic metabolic network (AMN).
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

from amn_core import KineticRegulationNet, StoichiometricConstraint, SensorReadoutHead

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"


class DAMN(nn.Module):
    """Dynamic Artificial Metabolic Network (dAMN) combining continuous CSTR ODEs with neural kinetics."""

    def __init__(
        self,
        num_sensors: int = 14,
        num_metabolites: int = 6,
        input_dim: int = 4,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        yield_glucose: float = 0.0811,
        dt_min: float = 5.0,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_metabolites = num_metabolites
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dt_min = dt_min

        # Pure neural sub-modules (no direct feedthrough)
        self.kinetic_net = KineticRegulationNet(
            num_metabolites=num_metabolites,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.stoich_constraint = StoichiometricConstraint(yield_biomass_on_glucose=yield_glucose)
        self.sensor_head = SensorReadoutHead(
            num_sensors=num_sensors,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        # Baseline media concentration vector (mmol/L)
        # Order: [glucose, ammonium, sulfate, sodium, chloride, uracil]
        self.register_buffer(
            "c_baseline",
            torch.tensor([194.28, 277.18, 20.82, 21.75, 21.16, 0.45], dtype=torch.float32),
        )

    def dynamics(
        self,
        biomass: torch.Tensor,
        volume: torch.Tensor,
        nutrients: torch.Tensor,
        inputs: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the CSTR time derivatives for all continuous state variables."""
        # Ensure positive physical quantities
        biomass = torch.clamp(biomass, min=1e-5)
        volume = torch.clamp(volume, min=1.0)
        nutrients = torch.clamp(nutrients, min=0.0)

        # Predict metabolic uptake fluxes (mmol/gDW/h), growth rate (1/h), and memory derivative
        # The kinetic network senses the internal/external concentrations and memory state
        v_uptake, mu_per_h, dh_dt = self.kinetic_net(biomass, volume, nutrients, latent)

        # Inputs enter strictly through the physical fluid flow terms:
        # inputs = [add_media, add_alt_media, remove_waste, dose_total]
        add_media = inputs[..., 0:1]
        add_alt = inputs[..., 1:2]
        remove_waste = inputs[..., 2:3]

        inflow_ml = add_media + add_alt
        outflow_ml = remove_waste

        # CSTR dilution rate on concentrations (1/min)
        dilution_per_min = inflow_ml / volume / self.dt_min

        # Volume rate of change (mL / min)
        d_volume = (inflow_ml - outflow_ml) / self.dt_min

        # Biomass rate of change: dX/dt = (mu/60 - Dilution) * X
        mu_per_min = mu_per_h / 60.0
        d_biomass = (mu_per_min - dilution_per_min) * biomass

        # Nutrient consumption rate (mmol / L / min)
        uptake_per_min = v_uptake * biomass / 60.0
        # Inflow replenishment rate assuming baseline broth feed (mmol / L / min)
        dose_rate = (inflow_ml / volume / self.dt_min) * self.c_baseline.unsqueeze(0)
        d_nutrients = dose_rate - uptake_per_min - (dilution_per_min * nutrients)

        return d_biomass, d_volume, d_nutrients, dh_dt, v_uptake, mu_per_h

    def rk4_step(
        self,
        biomass: torch.Tensor,
        volume: torch.Tensor,
        nutrients: torch.Tensor,
        inputs: torch.Tensor,
        latent: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """4th-order Runge-Kutta numerical integration step."""
        k1_x, k1_v, k1_c, k1_h, v1, mu1 = self.dynamics(biomass, volume, nutrients, inputs, latent)

        x_mid1 = biomass + 0.5 * dt * k1_x
        v_mid1 = volume + 0.5 * dt * k1_v
        c_mid1 = nutrients + 0.5 * dt * k1_c
        h_mid1 = latent + 0.5 * dt * k1_h
        k2_x, k2_v, k2_c, k2_h, _, _ = self.dynamics(x_mid1, v_mid1, c_mid1, inputs, h_mid1)

        x_mid2 = biomass + 0.5 * dt * k2_x
        v_mid2 = volume + 0.5 * dt * k2_v
        c_mid2 = nutrients + 0.5 * dt * k2_c
        h_mid2 = latent + 0.5 * dt * k2_h
        k3_x, k3_v, k3_c, k3_h, _, _ = self.dynamics(x_mid2, v_mid2, c_mid2, inputs, h_mid2)

        x_end = biomass + dt * k3_x
        v_end = volume + dt * k3_v
        c_end = nutrients + dt * k3_c
        h_end = latent + dt * k3_h
        k4_x, k4_v, k4_c, k4_h, _, _ = self.dynamics(x_end, v_end, c_end, inputs, h_end)

        next_x = biomass + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        next_v = volume + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        next_c = nutrients + (dt / 6.0) * (k1_c + 2 * k2_c + 2 * k3_c + k4_c)
        next_h = latent + (dt / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)

        return next_x, next_v, next_c, next_h, v1, mu1

    def forward(
        self,
        u_trajectory: torch.Tensor,
        init_biomass: torch.Tensor,
        init_volume: torch.Tensor,
        init_nutrients: torch.Tensor | None = None,
        init_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Integrate continuous dAMN forward through time for a batch of trajectories."""
        batch_size, time_steps, _ = u_trajectory.shape
        device = u_trajectory.device

        biomass = init_biomass
        volume = init_volume
        nutrients = init_nutrients if init_nutrients is not None else self.c_baseline.unsqueeze(0).repeat(batch_size, 1)
        latent = init_latent if init_latent is not None else torch.zeros(batch_size, self.latent_dim, device=device)

        y_preds = []
        mu_list = []
        fluxes_list = []
        biomass_list = []
        volume_list = []
        nutrients_list = []
        stoich_losses = []

        for t in range(time_steps):
            u_t = u_trajectory[:, t, :]

            # 1. Pure state sensor readout: decode physical/biological states ONLY
            _, mu_t, _ = self.kinetic_net(biomass, volume, nutrients, latent)
            y_pred_t = self.sensor_head(biomass, volume, mu_t, latent)
            y_preds.append(y_pred_t)

            # Record states
            biomass_list.append(biomass)
            volume_list.append(volume)
            nutrients_list.append(nutrients)
            mu_list.append(mu_t)

            # 2. Continuous dynamic RK4 integration step (dt = dt_min = 5.0 minutes)
            biomass, volume, nutrients, latent, v_t, _ = self.rk4_step(
                biomass, volume, nutrients, u_t, latent, dt=self.dt_min
            )
            fluxes_list.append(v_t)

            # 3. Compute stoichiometric violation penalty
            stoich_loss_t = self.stoich_constraint(mu_t, v_t)
            stoich_losses.append(stoich_loss_t)

        return {
            "y_pred": torch.stack(y_preds, dim=1),
            "biomass": torch.stack(biomass_list, dim=1),
            "volume": torch.stack(volume_list, dim=1),
            "nutrients": torch.stack(nutrients_list, dim=1),
            "growth_rate": torch.stack(mu_list, dim=1),
            "fluxes": torch.stack(fluxes_list, dim=1),
            "stoich_loss": torch.stack(stoich_losses).mean(),
        }
