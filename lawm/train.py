from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .model import LeastActionWorldModel
from .utils import (
    load_state_tensor,
    make_state_weights,
    make_time_grid,
    save_checkpoint,
    weighted_state_loss,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a physical-state Least-Action World Model")
    parser.add_argument("--train-pt", type=Path, required=True)
    parser.add_argument("--val-pt", type=Path, default=None)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--state-dim", type=int, default=9)
    parser.add_argument("--context-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--solver-iters", type=int, default=8)
    # 0.05 matches the LeastActionWorldModel default; the previous 0.25 here silently
    # trained a different (and numerically far less stable) solver than the model class.
    parser.add_argument("--solver-step-size", type=float, default=0.05)
    parser.add_argument(
        "--solver-grad",
        choices=("last_iterate", "unrolled"),
        default="last_iterate",
        help="'last_iterate' backpropagates through only the final DEL correction "
        "(identical forward, bounded backward); 'unrolled' is the paper-exact full "
        "unroll, which overflows fp32 on long rollouts (see tests/test_gradient_stability.py)",
    )
    parser.add_argument(
        "--bptt-grad-clip",
        type=float,
        default=10.0,
        help="per-rollout-step backward norm bound; the constant-velocity initialization "
        "grows gradients ~1.618^T without it. <=0 disables.",
    )
    parser.add_argument("--lambda-del", type=float, default=1e-2)
    parser.add_argument("--lambda-reg", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> LeastActionWorldModel:
    return LeastActionWorldModel(
        state_dim=args.state_dim,
        latent_dim=args.state_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        solver_iters=args.solver_iters,
        solver_step_size=args.solver_step_size,
        solver_grad=getattr(args, "solver_grad", "last_iterate"),
        bptt_grad_clip=getattr(args, "bptt_grad_clip", 10.0),
    ).to(device)


def batch_objective(
    model: LeastActionWorldModel,
    states: torch.Tensor,
    ts: torch.Tensor,
    weights: torch.Tensor,
    *,
    lambda_del: float,
    lambda_reg: float,
) -> Dict[str, torch.Tensor]:
    pred = model(states[:, 0], ts, state1=states[:, 1] if states.shape[1] > 1 else None)
    traj = weighted_state_loss(pred, states[:, : pred.shape[1]], weights)
    residual = model.dynamics.stationary_action_residual(model.q_sequence_from_state(pred), ts)
    del_loss = residual.square().mean() if residual.numel() else pred.new_tensor(0.0)
    reg_loss = model.regularization_loss(states[:, : pred.shape[1]])
    loss = traj + lambda_del * del_loss + lambda_reg * reg_loss
    return {"loss": loss, "traj": traj, "del": del_loss, "reg": reg_loss}


@torch.no_grad()
def evaluate(
    model: LeastActionWorldModel,
    states: torch.Tensor,
    ts: torch.Tensor,
    weights: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    totals = {"loss": 0.0, "traj": 0.0, "del": 0.0, "reg": 0.0}
    count = 0
    loader = DataLoader(TensorDataset(states), batch_size=args.batch_size, shuffle=False)
    for (batch,) in loader:
        batch = batch.to(device)
        with torch.enable_grad():
            out = batch_objective(
                model,
                batch,
                ts,
                weights,
                lambda_del=args.lambda_del,
                lambda_reg=args.lambda_reg,
            )
        count += batch.shape[0]
        for key in totals:
            totals[key] += float(out[key].detach().cpu()) * batch.shape[0]
    return {key: value / max(count, 1) for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_states = load_state_tensor(args.train_pt, args.state_dim, args.max_train_samples)
    val_states = load_state_tensor(args.val_pt, args.state_dim, args.max_val_samples) if args.val_pt else None

    ts = make_time_grid(train_states.shape[1], args.dt, device)
    weights = make_state_weights(args.state_dim, device)
    model = build_model(args, device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_states), batch_size=args.batch_size, shuffle=True)
    log_path = args.out_dir / "train_log.jsonl"

    with (args.out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)

    metrics: Dict[str, float] = {}
    skipped_nonfinite_total = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "traj": 0.0, "del": 0.0, "reg": 0.0}
        count = 0
        skipped_nonfinite = 0
        for (batch,) in loader:
            batch = batch.to(device)
            out = batch_objective(
                model,
                batch,
                ts,
                weights,
                lambda_del=args.lambda_del,
                lambda_reg=args.lambda_reg,
            )
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            # A single non-finite gradient step poisons every parameter, after which the
            # solver's sanitizers keep producing finite trajectories from a dead model
            # (traj then settles at the data's second moment and looks like convergence).
            # Never take that step; count it instead.
            grads_finite = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            if not grads_finite:
                skipped_nonfinite += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            count += batch.shape[0]
            for key in running:
                running[key] += float(out[key].detach().cpu()) * batch.shape[0]

        metrics = {f"train_{key}": value / max(count, 1) for key, value in running.items()}
        skipped_nonfinite_total += skipped_nonfinite
        metrics["skipped_nonfinite_steps"] = skipped_nonfinite
        if val_states is not None:
            val_ts = make_time_grid(val_states.shape[1], args.dt, device)
            val_metrics = evaluate(model, val_states, val_ts, weights, args, device)
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, **metrics}) + "\n")

        if epoch == 1 or epoch % args.save_every == 0 or epoch == args.epochs:
            print(json.dumps({"epoch": epoch, **metrics}, sort_keys=True), flush=True)
            save_checkpoint(args.out_dir / f"lawm_epoch{epoch:05d}.pth", model, optimizer, args=args, epoch=epoch, metrics=metrics)

    save_checkpoint(args.out_dir / "lawm_final.pth", model, optimizer, args=args, epoch=args.epochs, metrics=metrics)
    print(f"Saved final checkpoint to {args.out_dir / 'lawm_final.pth'}")
    if skipped_nonfinite_total:
        print(
            f"WARNING: skipped {skipped_nonfinite_total} optimizer step(s) with "
            "non-finite gradients; if this is more than a handful, reduce "
            "--solver-step-size or keep --solver-grad last_iterate."
        )


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
