"""Multimodal Dynamic Artificial Metabolic Network (dAMN) ODE integration engine.

Couples continuous CSTR fluid balances with neural-mechanistic metabolic kinetics,
thermal dynamics, UV irradiation, and 14-channel physical sensor decoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from amn_core import MultimodalKineticNet, StoichiometricConstraint, MultimodalSensorReadoutHead
except ImportError:
    from pioreactor.models.multimodal.damn.amn_core import MultimodalKineticNet, StoichiometricConstraint, MultimodalSensorReadoutHead


class MultimodalDAMN(nn.Module):
    """Continuous-time multimodal dAMN combining CSTR ODEs with neural-mechanistic kinetics."""

    def __init__(
        self,
        num_sensors: int = 14,
        num_metabolites: int = 6,
        input_dim: int = 6,
        latent_dim: int = 8,
        hidden_dim: int = 64,
        yield_glucose: float = 0.0811,
        yield_ammonium: float = 0.1519,
        dt_min: float = 5.0,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_metabolites = num_metabolites
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dt_min = dt_min

        self.kinetic_net = MultimodalKineticNet(
            num_metabolites=num_metabolites,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )
        self.stoich_constraint = StoichiometricConstraint(
            yield_biomass_on_glucose=yield_glucose,
            yield_biomass_on_ammonium=yield_ammonium,
        )
        self.sensor_head = MultimodalSensorReadoutHead(
            num_sensors=num_sensors,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        # Baseline broth concentrations (mmol/L): [glucose, ammonium, sulfate, sodium, chloride, uracil]
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
        """Compute the continuous CSTR time derivatives."""
        biomass = torch.clamp(biomass, min=1e-5)
        volume = torch.clamp(volume, min=1.0)
        nutrients = torch.clamp(nutrients, min=0.0)

        # Inputs: [add_media, add_alt_media, remove_waste, dose_total, temp_c, uv_intensity]
        add_media = inputs[..., 0:1]
        add_alt = inputs[..., 1:2]
        remove_waste = inputs[..., 2:3]
        temp_c = inputs[..., 4:5]
        uv_intensity = inputs[..., 5:6]

        inflow_ml = add_media + add_alt
        outflow_ml = remove_waste

        # 1. Neural-mechanistic kinetics under thermal and UV modulation
        v_uptake, mu_per_h, d_uv, d_thermal, dh_dt = self.kinetic_net(
            biomass, volume, nutrients, latent, temp_c, uv_intensity
        )

        # 2. CSTR physical fluid terms
        dilution_per_min = inflow_ml / volume / self.dt_min
        d_volume = (inflow_ml - outflow_ml) / self.dt_min

        # 3. Net specific growth rate in 1/min
        mu_net_per_min = (mu_per_h - d_uv - d_thermal) / 60.0
        d_biomass = (mu_net_per_min - dilution_per_min) * biomass

        # 4. Nutrient consumption and dosing replenishment
        uptake_per_min = (v_uptake * biomass) / 60.0
        dose_rate = (inflow_ml / volume / self.dt_min) * self.c_baseline.unsqueeze(0)
        d_nutrients = dose_rate - uptake_per_min - (dilution_per_min * nutrients)

        return d_biomass, d_volume, d_nutrients, dh_dt, mu_per_h, v_uptake

    def step(
        self,
        biomass: torch.Tensor,
        volume: torch.Tensor,
        nutrients: torch.Tensor,
        inputs: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance the continuous ODE system forward by one step dt_min via Euler integration."""
        d_biomass, d_volume, d_nutrients, dh_dt, mu_per_h, v_uptake = self.dynamics(
            biomass, volume, nutrients, inputs, latent
        )

        new_biomass = torch.clamp(biomass + d_biomass * self.dt_min, min=1e-5)
        new_volume = torch.clamp(volume + d_volume * self.dt_min, min=1.0)
        new_nutrients = torch.clamp(nutrients + d_nutrients * self.dt_min, min=0.0)
        new_latent = latent + dh_dt * self.dt_min

        return new_biomass, new_volume, new_nutrients, new_latent, mu_per_h, v_uptake

    def rollout(
        self,
        inputs: torch.Tensor,
        initial_od: torch.Tensor,
        initial_volume: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform an open-loop ODE rollout over T timesteps.
        
        Args:
            inputs: [B, T, 6] tensor of control inputs.
            initial_od: [B, 1] initial observed optical density.
            initial_volume: [B, 1] initial liquid volume in mL.
            
        Returns:
            pred_sensors: [B, T, 14] predicted sensor trajectory.
            pred_states: [B, T, 2 + num_metabolites + latent_dim] predicted continuous states.
            stoich_penalty: scalar mean stoichiometric yield penalty.
        """
        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Initial biomass from normalized OD (using 0.15 calibration)
        biomass = initial_od * 0.15
        volume = initial_volume.clone()
        nutrients = self.c_baseline.unsqueeze(0).repeat(batch_size, 1)
        latent = torch.zeros(batch_size, self.latent_dim, device=device)

        sensor_preds = []
        state_preds = []
        stoich_losses = []

        for t in range(seq_len):
            u_t = inputs[:, t, :]
            temp_c = u_t[:, 4:5]
            uv_intensity = u_t[:, 5:6]

            # Decode physical observations through readout head
            # mu is evaluated using current states
            _, mu_per_h, _, _, _ = self.kinetic_net(biomass, volume, nutrients, latent, temp_c, uv_intensity)
            sensors_t = self.sensor_head(biomass, volume, mu_per_h, latent, temp_c, uv_intensity)
            sensor_preds.append(sensors_t)

            state_t = torch.cat([biomass, volume, nutrients, latent], dim=-1)
            state_preds.append(state_t)

            # Integrate forward
            biomass, volume, nutrients, latent, mu_t, v_t = self.step(
                biomass, volume, nutrients, u_t, latent
            )

            stoich_loss = self.stoich_constraint(mu_t, v_t)
            stoich_losses.append(stoich_loss)

        pred_sensors = torch.stack(sensor_preds, dim=1)
        pred_states = torch.stack(state_preds, dim=1)
        stoich_penalty = torch.stack(stoich_losses).mean()

        return pred_sensors, pred_states, stoich_penalty
