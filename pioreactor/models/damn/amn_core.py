"""Artificial Metabolic Network (AMN) core layers implemented in PyTorch.

Pure neural-mechanistic formulation following Faulon et al.:
No direct algebraic feedthrough of control inputs.
Kinetic layers depend strictly on metabolic concentrations, biomass, volume, and latent state.
Sensor readout reflects purely the physical and biological state variables of the culture.
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
STOICH_DATA_PATH = SCRIPT_DIR / "core_stoichiometry.pt"


class KineticRegulationNet(nn.Module):
    """Neural kinetic layer predicting metabolic uptake fluxes, growth rate, and biological memory."""

    def __init__(
        self,
        num_metabolites: int = 6,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_metabolites = num_metabolites
        self.latent_dim = latent_dim

        # Input to neural kinetic network: [biomass(1), volume(1), nutrients(num_metabolites), latent_memory(latent_dim)]
        in_features = 1 + 1 + num_metabolites + latent_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output 1: Substrate uptake fluxes v(t) (mmol / gDW / h)
        # Softplus ensures non-negative uptake fluxes
        self.flux_head = nn.Sequential(
            nn.Linear(hidden_dim, num_metabolites),
            nn.Softplus(),
        )

        # Output 2: Specific growth rate mu(t) (1 / h)
        # Scaled by maximum biological capacity (~0.15 h^-1 max for room temp yeast)
        self.growth_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.max_mu = 0.15  # Maximum feasible specific growth rate (h^-1)

        # Output 3: Time derivative of latent biological memory dh/dt
        self.memory_head = nn.Linear(hidden_dim, latent_dim)
        self.tau_memory = nn.Parameter(torch.ones(latent_dim) * 30.0)  # Memory time constant (~30 min)

    def forward(
        self,
        biomass: torch.Tensor,
        volume: torch.Tensor,
        nutrients: torch.Tensor,
        latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute instantaneous metabolic fluxes, specific growth rate, and latent state derivative."""
        features = torch.cat([biomass, volume, nutrients, latent], dim=-1)
        hidden = self.net(features)

        # Uptake fluxes (mmol / gDW / h)
        v_uptake = self.flux_head(hidden)

        # Substrate availability soft mask: if nutrient approaches 0, uptake must approach 0
        nutrient_availability = torch.clamp(nutrients / (nutrients + 0.05), 0.0, 1.0)
        v_uptake = v_uptake * nutrient_availability

        # Growth rate (1 / h)
        mu = self.growth_head(hidden) * self.max_mu

        # Memory derivative with leaky biological relaxation: dh/dt = f(z) - h / tau
        tau = torch.clamp(self.tau_memory, min=5.0, max=120.0)
        dh_dt = torch.tanh(self.memory_head(hidden)) - (latent / tau)

        return v_uptake, mu, dh_dt


class StoichiometricConstraint(nn.Module):
    """Enforces stoichiometric mass balance linking growth to substrate consumption."""

    def __init__(self, yield_biomass_on_glucose: float = 0.0811):
        super().__init__()
        self.yield_glucose = yield_biomass_on_glucose

    def forward(self, mu: torch.Tensor, v_uptake: torch.Tensor) -> torch.Tensor:
        """Compute stoichiometric violation penalty: ||mu - Yield * v_glucose||^2."""
        glucose_uptake = v_uptake[..., 0:1]
        stoich_diff = mu - (self.yield_glucose * glucose_uptake)
        penalty = torch.mean(stoich_diff ** 2)
        return penalty


class SensorReadoutHead(nn.Module):
    """Pure state decoder: decodes physical culture states into all 14 sensor dimensions.
    
    NO direct feedthrough of exogenous pump inputs u(t).
    Sensors reflect purely the internal state of the culture: biomass, volume, growth, and latent state.
    """

    def __init__(
        self,
        num_sensors: int = 14,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        # Pure state inputs: biomass(1), volume(1), growth_rate(1), latent_memory(latent_dim)
        in_features = 1 + 1 + 1 + latent_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_sensors),
        )

    def forward(
        self,
        biomass: torch.Tensor,
        volume: torch.Tensor,
        mu: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Predict normalized sensor vector strictly from physical/biological states."""
        features = torch.cat([biomass, volume, mu, latent], dim=-1)
        y_pred = self.net(features)
        return y_pred
