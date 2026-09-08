"""Multimodal Artificial Metabolic Network (dAMN) core layers in PyTorch.

Extends the neural-mechanistic formulation to full multimodal inputs:
- Chemical dosing (carbon, nitrogen, sulfur, salt, uracil)
- Temperature modulation (thermal kinetics & stress)
- UV irradiation (photo-inactivation & DNA repair dynamics)
"""

from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_TEMP_C = 30.0


class MultimodalKineticNet(nn.Module):
    """Neural kinetic layer predicting metabolic uptake fluxes, growth rate, and latent biological memory."""

    def __init__(
        self,
        num_metabolites: int = 6,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_metabolites = num_metabolites
        self.latent_dim = latent_dim

        # Input: biomass(1), volume(1), nutrients(num_metabolites), latent_memory(latent_dim), delta_T(1), uv_intensity(1)
        in_features = 1 + 1 + num_metabolites + latent_dim + 1 + 1

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Output 1: Substrate uptake fluxes v(t) (mmol / gDW / h)
        self.flux_head = nn.Sequential(
            nn.Linear(hidden_dim, num_metabolites),
            nn.Softplus(),
        )

        # Output 2: Specific growth rate mu(t) (1 / h)
        self.growth_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.max_mu = 0.20  # Maximum feasible specific growth rate (h^-1)

        # Output 3: Environmental stress mortality rates (UV and heat)
        self.uv_decay_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )
        self.thermal_decay_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        # Output 4: Time derivative of latent biological memory dh/dt
        self.memory_head = nn.Linear(hidden_dim, latent_dim)
        self.tau_memory = nn.Parameter(torch.ones(latent_dim) * 30.0)

    def forward(
        self,
        biomass: torch.Tensor,
        volume: torch.Tensor,
        nutrients: torch.Tensor,
        latent: torch.Tensor,
        temp_c: torch.Tensor,
        uv_intensity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Normalize environmental conditions around baselines
        delta_t = (temp_c - BASELINE_TEMP_C) / 10.0  # ~ [-1.0, +1.0]
        norm_uv = uv_intensity / 100.0              # ~ [0.0, 1.0]

        features = torch.cat([biomass, volume, nutrients, latent, delta_t, norm_uv], dim=-1)
        hidden = self.net(features)

        # Uptake fluxes (mmol / gDW / h)
        v_uptake = self.flux_head(hidden)
        nutrient_availability = torch.clamp(nutrients / (nutrients + 0.05), 0.0, 1.0)
        v_uptake = v_uptake * nutrient_availability

        # Growth rate (1 / h)
        mu = self.growth_head(hidden) * self.max_mu

        # Stress death rates (1 / h)
        d_uv = self.uv_decay_head(hidden) * norm_uv * 0.20
        d_thermal = self.thermal_decay_head(hidden) * torch.clamp(delta_t, min=0.0) ** 2 * 0.10

        # Memory derivative with adaptive leaky relaxation: dh/dt = tanh(NN) - h/tau
        tau = torch.clamp(self.tau_memory, min=5.0, max=120.0)
        dh_dt = torch.tanh(self.memory_head(hidden)) - (latent / tau)

        return v_uptake, mu, d_uv, d_thermal, dh_dt


class StoichiometricConstraint(nn.Module):
    """Enforces stoichiometric mass balance linking growth to substrate consumption."""

    def __init__(
        self,
        yield_biomass_on_glucose: float = 0.0811,
        yield_biomass_on_ammonium: float = 0.1519,
        stoich_pt_path: Path | str | None = None,
    ):
        super().__init__()
        self.yield_glucose = yield_biomass_on_glucose
        self.yield_ammonium = yield_biomass_on_ammonium
        self.metabolites: list[str] = []
        self.reactions: list[str] = []

        if stoich_pt_path is None:
            stoich_pt_path = SCRIPT_DIR / "artifacts" / "stoichiometry" / "core_stoichiometry.pt"

        path = Path(stoich_pt_path)
        if path.exists():
            data = torch.load(path, map_location="cpu", weights_only=False)
            if "S_core" in data and data["S_core"].numel() > 0:
                self.register_buffer("S_core", data["S_core"])
                self.metabolites = data.get("metabolites", [])
                self.reactions = data.get("reactions", [])
            else:
                self.register_buffer("S_core", None)
        else:
            self.register_buffer("S_core", None)

    def debug_print_stoichiometry(self) -> None:
        """Print debug statement showing stoichiometric matrix and parameters."""
        print("[DEBUG StoichiometricConstraint]")
        print(f"  * Biomass-on-glucose yield Y_x/glc: {self.yield_glucose:.4f} gDW/mmol")
        print(f"  * Biomass-on-ammonium yield Y_x/nh4: {self.yield_ammonium:.4f} gDW/mmol")
        if self.S_core is not None:
            print(f"  * S_core tensor shape: {tuple(self.S_core.shape)}")
            print(f"  * Metabolite IDs ({len(self.metabolites)}): {self.metabolites}")
            print(f"  * Reaction IDs   ({len(self.reactions)}): {self.reactions}")
            print(f"  * S_core values:\n{self.S_core.cpu().numpy()}")
        else:
            print("  * S_core tensor: None (macroscopic yield constraint active)")

    def forward(self, mu: torch.Tensor, v_uptake: torch.Tensor) -> torch.Tensor:
        """Compute stoichiometric violation penalty: linking growth rate to nutrient consumption."""
        glucose_uptake = v_uptake[..., 0:1]
        nh4_uptake = v_uptake[..., 1:2]

        diff_glc = mu - (self.yield_glucose * glucose_uptake)
        diff_nh4 = mu - (self.yield_ammonium * nh4_uptake)

        return torch.mean(diff_glc ** 2) + 0.5 * torch.mean(diff_nh4 ** 2)


class MultimodalSensorReadoutHead(nn.Module):
    """Pure state decoder: decodes physical culture states into all 14 sensor dimensions."""

    def __init__(
        self,
        num_sensors: int = 14,
        latent_dim: int = 8,
        hidden_dim: int = 64,
    ):
        super().__init__()
        # State inputs: biomass(1), volume(1), growth_rate(1), latent_memory(latent_dim), delta_T(1), norm_uv(1)
        in_features = 1 + 1 + 1 + latent_dim + 1 + 1

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
        temp_c: torch.Tensor,
        uv_intensity: torch.Tensor,
    ) -> torch.Tensor:
        delta_t = (temp_c - BASELINE_TEMP_C) / 10.0
        norm_uv = uv_intensity / 100.0
        features = torch.cat([biomass, volume, mu, latent, delta_t, norm_uv], dim=-1)
        return self.net(features)
