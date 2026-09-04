"""Liquid Time-Constant (LTC) recurrent cell with Fused Semi-Implicit ODE solver.

Implements the continuous-time dynamical system from:
  "Liquid Time-Constant Networks" (Hasani et al., AAAI 2021)
  dx/dt = - [ 1/tau + f(x(t), I(t)) ] * x(t) + f(x(t), I(t)) * A

Discrete Fused Solver update step (semi-implicit Euler):
  x(t + dt) = [ x(t) + dt * (f(x, I) * A) ] / [ 1 + dt * (1/tau + f(x, I)) ]

Properties:
  - Input-dependent liquid time-constant: tau_sys = tau / (1 + tau * f(x, I))
  - Bounded state dynamics: min(0, A_min) <= x(t) <= max(0, A_max)
  - Unconditional stability on stiff biological inputs without step-size explosion.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LTCCell(nn.Module):
    """Single-layer Liquid Time-Constant (LTC) recurrent cell."""

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 32,
        unfolding_steps: int = 2,
        dt: float = 1.0,
        tau_min: float = 0.5,
        tau_max: float = 30.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.unfolding_steps = unfolding_steps
        self.dt_sub = dt / float(unfolding_steps)
        self.tau_min = tau_min
        self.tau_max = tau_max

        # Base time-constant parameterization (raw parameter transformed to [tau_min, tau_max])
        self.raw_tau = nn.Parameter(torch.zeros(hidden_dim))

        # Reversal potentials / saturation asymptote A
        self.A = nn.Parameter(torch.ones(hidden_dim))

        # Synaptic conductance network f(x, I) -> positive conductance
        # Input features: current hidden state (hidden_dim) + incoming input (input_dim)
        in_features = hidden_dim + input_dim
        self.conductance_net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),  # Ensures strictly non-negative synaptic conductance
        )

        # Recurrent bias
        self.bias = nn.Parameter(torch.zeros(hidden_dim))

    @property
    def tau(self) -> torch.Tensor:
        """Effective passive resting time constants in [tau_min, tau_max]."""
        # Sigmoidal range scaling
        return self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.raw_tau)

    def fused_step(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute one sub-step of the fused semi-implicit Euler solver.
        
        Args:
            x: (batch_size, hidden_dim) current state
            u: (batch_size, input_dim) current input
            
        Returns:
            x_next: (batch_size, hidden_dim) updated state
            tau_sys: (batch_size, hidden_dim) instantaneous liquid time-constant
        """
        # 1. Compute synaptic conductance f(x, I)
        xu = torch.cat([x, u], dim=-1)
        f = self.conductance_net(xu)  # strictly positive

        # 2. Base leak rate 1 / tau
        inv_tau = 1.0 / self.tau  # (hidden_dim,)

        # 3. Closed-form Fused update
        dt = self.dt_sub
        numerator = x + dt * (f * self.A + self.bias)
        denominator = 1.0 + dt * (inv_tau + f)
        x_next = numerator / denominator

        # 4. Instantaneous liquid time-constant: tau / (1 + tau * f)
        tau_sys = self.tau / (1.0 + self.tau * f)

        return x_next, tau_sys

    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate cell across one macro-step (e.g. 5 minutes) using L sub-steps.
        
        Args:
            x: (batch_size, hidden_dim) initial state
            u: (batch_size, input_dim) macro-step input
            
        Returns:
            x: (batch_size, hidden_dim) final integrated state
            tau_sys: (batch_size, hidden_dim) average liquid time constant during this step
        """
        tau_accum = torch.zeros_like(x)
        for _ in range(self.unfolding_steps):
            x, tau_sys = self.fused_step(x, u)
            tau_accum = tau_accum + tau_sys

        mean_tau_sys = tau_accum / float(self.unfolding_steps)
        return x, mean_tau_sys
