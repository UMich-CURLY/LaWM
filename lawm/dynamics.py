from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from .lagrangian import LatentDiscreteLagrangian, mlp


@dataclass
class DELSolveInfo:
    residual_norm: torch.Tensor
    iterations: int
    sanitizer_hits: int = 0


class _BackwardNormClip(torch.autograd.Function):
    """Identity in the forward pass; clips the gradient norm in the backward pass.

    Inserted at each rollout-step boundary. The constant-velocity initialization
    ``q_next = 2 q_curr - q_prev`` has companion-matrix spectral radius
    (1 + sqrt(5)) / 2 ~ 1.618, so backpropagating a T-step rollout multiplies gradients
    by ~1.618^T regardless of the data or the solver (T=62 on the toy example is ~1e13,
    which overflows fp32 into inf/NaN). Bounding the norm once per crossing turns that
    geometric growth into a bounded flow while leaving the forward rollout untouched.
    Non-finite incoming gradients are zeroed before rescaling so an upstream overflow
    cannot poison every earlier step.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, max_norm: float) -> torch.Tensor:
        ctx.max_norm = float(max_norm)
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        if grad is None:
            return None, None
        norm = grad.norm()
        if torch.isfinite(norm) and float(norm) <= ctx.max_norm:
            return grad, None
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        norm = grad.norm().clamp_min(1e-12)
        return grad * (ctx.max_norm / norm), None


class LatentVariationalDynamics(nn.Module):
    """Least-action latent transition induced by the DEL condition."""

    def __init__(
        self,
        q_dim: int,
        context_dim: int = 16,
        hidden_dim: int = 128,
        depth: int = 3,
        solver_iters: int = 8,
        solver_step_size: float = 0.05,
        del_residual_scale: float = 1.0,
        solver_grad: str = "last_iterate",
        bptt_grad_clip: float = 10.0,
    ) -> None:
        super().__init__()
        if solver_grad not in {"last_iterate", "unrolled"}:
            raise ValueError("solver_grad must be 'last_iterate' or 'unrolled'")
        self.q_dim = int(q_dim)
        self.context_dim = int(context_dim)
        self.solver_iters = int(solver_iters)
        self.solver_step_size = float(solver_step_size)
        self.del_residual_scale = float(del_residual_scale)
        # 'last_iterate': the first N-1 residual corrections run without a graph and only
        # the final correction is differentiable (the standard truncated/implicit gradient
        # for an unrolled root solve). Forward results are identical to 'unrolled'; the
        # within-step gradient chain of length N (which compounds with the rollout-length
        # chain) is reduced to length 1. 'unrolled' restores full backprop through every
        # solver iteration (the exact formulation of Eq. 20 in the paper).
        self.solver_grad = solver_grad
        # <= 0 disables the per-step backward norm clip.
        self.bptt_grad_clip = float(bptt_grad_clip)
        self.context_net = (
            mlp(3 * self.q_dim, hidden_dim, self.context_dim, depth)
            if self.context_dim > 0
            else None
        )
        self.lagrangian = LatentDiscreteLagrangian(
            q_dim=self.q_dim,
            context_dim=self.context_dim,
            hidden_dim=hidden_dim,
            depth=depth,
        )

    def infer_context(self, q_context: torch.Tensor) -> Optional[torch.Tensor]:
        if self.context_net is None:
            return None
        if q_context.shape[1] == 1:
            q_prev = q_context[:, 0]
            q_last = q_context[:, 0]
        else:
            q_prev = q_context[:, -2]
            q_last = q_context[:, -1]
        dq = q_last - q_prev
        return self.context_net(torch.cat([q_prev, q_last, dq], dim=-1))

    def _h_at(self, ts: torch.Tensor, idx: int, fallback: torch.Tensor) -> torch.Tensor:
        if ts.numel() <= 1:
            return fallback
        left = max(0, min(idx, ts.numel() - 2))
        return (ts[left + 1] - ts[left]).abs().clamp_min(1e-8)

    def del_residual(
        self,
        q_prev: torch.Tensor,
        q_curr: torch.Tensor,
        q_next: torch.Tensor,
        h_prev: torch.Tensor | float,
        h_next: Optional[torch.Tensor | float] = None,
        eta: Optional[torch.Tensor] = None,
        create_graph: bool = True,
    ) -> torch.Tensor:
        if h_next is None:
            h_next = h_prev
        q_prev_req = q_prev.detach().requires_grad_(True)
        q_curr_req = q_curr.detach().requires_grad_(True)
        q_next_req = q_next.requires_grad_(True)

        left = self.lagrangian.discrete_lagrangian(q_prev_req, q_curr_req, h_prev, eta).sum()
        right = self.lagrangian.discrete_lagrangian(q_curr_req, q_next_req, h_next, eta).sum()
        d2_left = torch.autograd.grad(left, q_curr_req, create_graph=create_graph, retain_graph=True)[0]
        d1_right = torch.autograd.grad(right, q_curr_req, create_graph=create_graph, retain_graph=True)[0]
        return self.del_residual_scale * (d2_left + d1_right)

    def step(
        self,
        q_prev: torch.Tensor,
        q_curr: torch.Tensor,
        h_prev: torch.Tensor | float,
        h_next: Optional[torch.Tensor | float],
        eta: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, DELSolveInfo]:
        iters = max(self.solver_iters, 1)
        q_init = q_curr + (q_curr - q_prev)
        if self.bptt_grad_clip > 0:
            # Bound the through-time gradient once per rollout step; see _BackwardNormClip.
            q_init = _BackwardNormClip.apply(q_init, self.bptt_grad_clip)
        mass = self.lagrangian.mass_diag(q_curr, eta).detach().clamp_min(1e-4)
        sanitizer_hits = 0

        def correct(q_next: torch.Tensor, create_graph: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
            residual = self.del_residual(
                q_prev, q_curr, q_next, h_prev, h_next, eta, create_graph=create_graph
            )
            hits = int((~torch.isfinite(residual)).sum())
            residual = torch.nan_to_num(residual, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
            update = (self.solver_step_size * residual / mass).clamp(-1.0, 1.0)
            q_out = q_next - update
            hits += int((~torch.isfinite(q_out)).sum())
            q_out = torch.nan_to_num(q_out, nan=0.0, posinf=1e4, neginf=-1e4)
            return q_out, residual, hits

        if self.solver_grad == "unrolled":
            q_next = q_init
            residual = q_init.new_zeros(q_init.shape)
            for _ in range(iters):
                q_next, residual, hits = correct(q_next, create_graph=True)
                sanitizer_hits += hits
        else:
            # last_iterate: iterate to (near) the DEL root without a graph, then take one
            # differentiable correction from the detached iterate. The correction is added
            # to the LIVE q_init so through-time credit still flows along the
            # initialization path; the solver contributes a single-iteration gradient.
            with torch.enable_grad():
                q_iter = q_init.detach()
                for _ in range(iters - 1):
                    q_iter, _, hits = correct(q_iter, create_graph=False)
                    q_iter = q_iter.detach()
                    sanitizer_hits += hits
                q_solved, residual, hits = correct(q_iter, create_graph=True)
                sanitizer_hits += hits
            q_next = q_init + (q_solved - q_init.detach())
        info = DELSolveInfo(
            residual_norm=residual.norm(dim=-1).mean(),
            iterations=iters,
            sanitizer_hits=sanitizer_hits,
        )
        return q_next, info

    def rollout(
        self,
        q0: torch.Tensor,
        q1: torch.Tensor,
        ts: torch.Tensor,
        eta: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if ts.ndim != 1:
            raise ValueError(f"ts must have shape (T,), got {tuple(ts.shape)}")
        if ts.numel() < 2:
            return {"q": q0[:, None], "del_residual": q0.new_tensor(0.0)}

        qs = [q0, q1]
        residuals = []
        sanitizer_hits = 0
        fallback_h = (ts[1] - ts[0]).abs().clamp_min(1e-8)
        for k in range(1, ts.numel() - 1):
            h_prev = self._h_at(ts, k - 1, fallback_h)
            h_next = self._h_at(ts, k, fallback_h)
            q_next, info = self.step(qs[-2], qs[-1], h_prev, h_next, eta)
            qs.append(q_next)
            residuals.append(info.residual_norm)
            sanitizer_hits += info.sanitizer_hits
        residual = torch.stack(residuals).mean() if residuals else q0.new_tensor(0.0)
        return {
            "q": torch.stack(qs[: ts.numel()], dim=1),
            "del_residual": residual,
            "sanitizer_hits": q0.new_tensor(float(sanitizer_hits)),
        }

    def stationary_action_residual(self, q_seq: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        if q_seq.shape[1] < 3:
            return q_seq.new_zeros(q_seq.shape[0], 0, self.q_dim)
        eta = self.infer_context(q_seq[:, :2])
        fallback_h = (ts[1] - ts[0]).abs().clamp_min(1e-8) if ts.numel() > 1 else q_seq.new_tensor(1.0)
        residuals = []
        for k in range(1, q_seq.shape[1] - 1):
            h_prev = self._h_at(ts, k - 1, fallback_h)
            h_next = self._h_at(ts, k, fallback_h)
            residuals.append(
                self.del_residual(
                    q_seq[:, k - 1],
                    q_seq[:, k],
                    q_seq[:, k + 1],
                    h_prev,
                    h_next,
                    eta,
                    create_graph=True,
                )
            )
        return torch.stack(residuals, dim=1)
