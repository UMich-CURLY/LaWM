"""Gradient stability of the DEL rollout backward pass.

Reproduces and guards against two compounding gradient explosions in training:

1. The constant-velocity initialization ``q_next = 2 q_curr - q_prev`` has
   companion-matrix spectral radius (1 + sqrt(5)) / 2 ~ 1.618, so full backprop
   through a T-step rollout scales gradients by ~1.618^T (T = 62 interior steps on
   ``examples/toy_parabolic.py`` gives ~1e13 before any solver amplification).
2. Backprop through all N unrolled DEL corrections (each residual keeps a live,
   second-order dependence on the running iterate) multiplies that further.

On fp32 this overflows to inf/NaN on the repo's own toy example at the previous
train-script defaults; ``dynamics.step``'s nan_to_num sanitizers then keep producing
finite trajectories from the poisoned parameters, so the trajectory loss silently
settles at the data's second moment and imitates convergence.

The defaults under test: ``solver_grad='last_iterate'`` (identical forward, one
differentiable correction) plus a per-rollout-step backward norm clip.

Run: ``pytest tests/test_gradient_stability.py`` (CPU, ~1 minute).
"""
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, PROJECT_ROOT / "examples"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lawm.model import LeastActionWorldModel
from lawm.utils import make_state_weights, make_time_grid, weighted_state_loss
from toy_parabolic import make_toy_parabolic


def make_synth_cruise(samples: int, steps: int, dt: float, seed: int) -> torch.Tensor:
    """Planar cruising tracks: random speed, gentle constant yaw rate. 8-dim state
    [x, y, cos(yaw), sin(yaw), vx, vy, length, width] in a start-aligned frame."""
    g = torch.Generator().manual_seed(seed)
    speed = torch.empty(samples, 1).uniform_(1.5, 3.0, generator=g)
    yaw_rate = torch.empty(samples, 1).uniform_(-0.15, 0.15, generator=g)
    t = torch.arange(steps, dtype=torch.float32)[None, :] * dt
    yaw = yaw_rate * t
    vx = speed * torch.cos(yaw)
    vy = speed * torch.sin(yaw)
    x = torch.cumsum(torch.cat([torch.zeros(samples, 1), vx[:, :-1] * dt], dim=1), dim=1)
    y = torch.cumsum(torch.cat([torch.zeros(samples, 1), vy[:, :-1] * dt], dim=1), dim=1)
    length = torch.empty(samples, 1).uniform_(0.4, 0.9, generator=g).expand(-1, steps)
    width = torch.empty(samples, 1).uniform_(0.15, 0.3, generator=g).expand(-1, steps)
    return torch.stack([x, y, torch.cos(yaw), torch.sin(yaw), vx, vy, length, width], dim=-1)


def first_backward_grad_stats(states: torch.Tensor, dt: float, **model_kwargs):
    state_dim = states.shape[-1]
    ts = make_time_grid(states.shape[1], dt, "cpu")
    weights = make_state_weights(state_dim, "cpu")
    torch.manual_seed(0)
    model = LeastActionWorldModel(state_dim=state_dim, latent_dim=state_dim, **model_kwargs)
    pred = model(states[:, 0], ts, state1=states[:, 1])
    loss = weighted_state_loss(pred, states[:, : pred.shape[1]], weights)
    model.zero_grad()
    loss.backward()
    nonfinite = 0
    max_grad = 0.0
    for _, p in model.named_parameters():
        if p.grad is None:
            continue
        if torch.isfinite(p.grad).all():
            max_grad = max(max_grad, float(p.grad.abs().max()))
        else:
            nonfinite += 1
    return nonfinite, max_grad, model, pred


def test_defaults_are_stable_on_toy_parabolic():
    states = make_toy_parabolic(16, 64, 0.02, 0)
    nonfinite, max_grad, _, _ = first_backward_grad_stats(states, 0.02)
    assert nonfinite == 0, f"{nonfinite} parameter tensors carry non-finite gradients"
    assert max_grad < 1e6, f"gradients still exploding: max |grad| = {max_grad:.3g}"


def test_defaults_are_stable_on_synthetic_cruise():
    states = make_synth_cruise(16, 16, 0.1, 0)
    nonfinite, max_grad, _, _ = first_backward_grad_stats(states, 0.1)
    assert nonfinite == 0, f"{nonfinite} parameter tensors carry non-finite gradients"
    assert max_grad < 1e6, f"gradients still exploding: max |grad| = {max_grad:.3g}"


def test_unrolled_full_bptt_explodes_documenting_the_bug():
    """The paper-exact full unroll without the backward clip overflows on the repo's own
    toy data. This test documents the failure mode the defaults exist to prevent; if it
    ever starts passing, the defaults can be revisited."""
    states = make_toy_parabolic(16, 64, 0.02, 0)
    nonfinite, max_grad, _, _ = first_backward_grad_stats(
        states, 0.02, solver_grad="unrolled", bptt_grad_clip=0.0
    )
    assert nonfinite > 0 or max_grad > 1e12


def test_last_iterate_forward_matches_unrolled():
    """solver_grad changes only the backward pass: identical rollouts forward."""
    states = make_synth_cruise(8, 12, 0.1, 1)
    ts = make_time_grid(states.shape[1], 0.1, "cpu")
    preds = []
    for mode in ("last_iterate", "unrolled"):
        torch.manual_seed(0)
        model = LeastActionWorldModel(
            state_dim=8, latent_dim=8, solver_grad=mode, bptt_grad_clip=10.0
        )
        # The DEL solve differentiates the Lagrangian internally, so the rollout needs
        # grad machinery even at inference (same convention as lawm.train.evaluate).
        with torch.enable_grad():
            preds.append(model(states[:, 0], ts, state1=states[:, 1]).detach())
    assert torch.allclose(preds[0], preds[1], atol=1e-6), (
        f"forward drift between gradient modes: max delta "
        f"{(preds[0] - preds[1]).abs().max():.3g}"
    )


def test_short_training_learns_with_finite_terms():
    """A short real training on the synthetic cruise must (a) keep every loss term
    finite, (b) beat both the trivial second-moment baseline (the signature of the old
    sanitizer-masked collapse) and the constant-velocity extrapolation baseline."""
    from lawm.train import batch_objective

    states = make_synth_cruise(32, 16, 0.1, 2)
    ts = make_time_grid(states.shape[1], 0.1, "cpu")
    weights = make_state_weights(states.shape[-1], "cpu")
    torch.manual_seed(0)
    model = LeastActionWorldModel(state_dim=8, latent_dim=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    last = None
    for _ in range(150):
        out = batch_objective(model, states, ts, weights, lambda_del=1e-2, lambda_reg=1e-4)
        for key, value in out.items():
            assert torch.isfinite(value), f"{key} went non-finite during training"
        optimizer.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = out
    second_moment = float(states.square().mean())
    cv = states.clone()
    for k in range(2, states.shape[1]):
        cv[:, k] = 2 * cv[:, k - 1] - cv[:, k - 2]
    cv_baseline = float(weighted_state_loss(cv, states, weights))
    final_traj = float(last["traj"])
    assert final_traj < 0.5 * second_moment, (
        f"traj {final_traj:.4g} is at the trivial second-moment baseline "
        f"{second_moment:.4g} -- the collapsed-model signature"
    )
    assert final_traj < cv_baseline, (
        f"traj {final_traj:.4g} does not beat constant-velocity extrapolation "
        f"{cv_baseline:.4g}"
    )
