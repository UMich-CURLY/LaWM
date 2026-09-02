from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .dynamics import LatentVariationalDynamics


class LeastActionWorldModel(nn.Module):
    """Physical-state LaWM with a DEL-induced latent rollout."""

    def __init__(
        self,
        state_dim: int = 9,
        latent_dim: Optional[int] = None,
        context_dim: int = 16,
        hidden_dim: int = 128,
        depth: int = 3,
        solver_iters: int = 8,
        solver_step_size: float = 0.05,
        solver_grad: str = "last_iterate",
        bptt_grad_clip: float = 10.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim or state_dim)
        if self.latent_dim != self.state_dim:
            raise ValueError("This state-space implementation requires latent_dim == state_dim")
        self.context_dim = int(context_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.solver_iters = int(solver_iters)
        self.solver_step_size = float(solver_step_size)
        self.dynamics = LatentVariationalDynamics(
            q_dim=self.latent_dim,
            context_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            depth=self.depth,
            solver_iters=self.solver_iters,
            solver_step_size=self.solver_step_size,
            solver_grad=solver_grad,
            bptt_grad_clip=bptt_grad_clip,
        )

    def q_sequence_from_state(self, states: torch.Tensor) -> torch.Tensor:
        return states[..., : self.latent_dim]

    def state_from_q_sequence(self, q_seq: torch.Tensor) -> torch.Tensor:
        return q_seq

    def _default_second_state(self, z0: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        h = (ts[1] - ts[0]).abs().clamp_min(1e-8) if ts.numel() > 1 else z0.new_tensor(1.0)
        dz = torch.zeros_like(z0)
        if self.state_dim >= 4:
            dz[:, 0] = z0[:, 2]
            dz[:, 1] = z0[:, 3]
        if self.state_dim >= 6:
            dz[:, 4] = z0[:, 5]
        return z0 + h * dz

    def forward(
        self,
        z0: torch.Tensor,
        ts: torch.Tensor,
        *,
        state1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if state1 is None:
            state1 = self._default_second_state(z0, ts)
        q_context = torch.stack([self.q_sequence_from_state(z0), self.q_sequence_from_state(state1)], dim=1)
        eta = self.dynamics.infer_context(q_context)
        out = self.dynamics.rollout(q_context[:, 0], q_context[:, 1], ts, eta=eta)
        return self.state_from_q_sequence(out["q"])

    def forward_from_context(self, context: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if context.ndim != 3:
            raise ValueError(f"context must have shape (B,L,D), got {tuple(context.shape)}")
        if context.shape[1] == 1:
            q0 = self.q_sequence_from_state(context[:, 0])
            q1 = self._default_second_state(context[:, 0], ts)
            q_context = torch.stack([q0, q1], dim=1)
        else:
            q_context = self.q_sequence_from_state(context[:, -2:])
        eta = self.dynamics.infer_context(q_context)
        out = self.dynamics.rollout(q_context[:, 0], q_context[:, 1], ts, eta=eta)
        return self.state_from_q_sequence(out["q"])

    def total_energy_from_state(self, states: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        q = self.q_sequence_from_state(states)
        if q.shape[1] < 2:
            v = torch.zeros_like(q)
        else:
            h = (ts[1] - ts[0]).abs().clamp_min(1e-8) if ts.numel() > 1 else q.new_tensor(1.0)
            v = torch.zeros_like(q)
            if q.shape[1] > 2:
                v[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2.0 * h)
            v[:, 0] = (q[:, 1] - q[:, 0]) / h
            v[:, -1] = (q[:, -1] - q[:, -2]) / h
        eta = self.dynamics.infer_context(q[:, :2]) if q.shape[1] >= 2 else None
        flat_q = q.reshape(-1, q.shape[-1])
        flat_v = v.reshape(-1, v.shape[-1])
        if eta is not None:
            eta = eta[:, None].expand(-1, q.shape[1], -1).reshape(-1, eta.shape[-1])
        mass = self.dynamics.lagrangian.mass_diag(flat_q, eta)
        potential = self.dynamics.lagrangian.potential(flat_q, eta)
        energy = 0.5 * (mass * flat_v.square()).sum(dim=-1) + potential
        return energy.reshape(q.shape[:2])

    def regularization_loss(self, states: torch.Tensor) -> torch.Tensor:
        q = self.q_sequence_from_state(states)
        eta = self.dynamics.infer_context(q[:, :2]) if q.shape[1] >= 2 else None
        if eta is not None:
            eta = eta[:, None].expand(-1, q.shape[1], -1)
        return self.dynamics.lagrangian.mass_conditioning_loss(q, eta)
