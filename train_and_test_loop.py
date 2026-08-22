# Training/testing loop (roadmap step 2).
#
# Trains ChessNet's policy + value heads jointly on the (input, policy_idx,
# value) triples produced by pgn_pipeline.py's PGNIterableDataset:
#   loss = cross_entropy(policy_logits, policy_idx) + mse(value_pred, value_target)
# using AdamW + a cosine LR schedule, with an AMP-accelerated forward pass
# (matching network_model.py's amp_context()) and a matching GradScaler for
# the backward/optimizer step on CUDA.
#
# Metrics reported each epoch:
#   policy_loss, value_loss, total_loss   -- what's actually optimized
#   policy_top1_acc  -- how often the network's raw argmax move matches the
#                        human move played (unmasked by legality -- this is
#                        a supervised-learning sanity metric, not a search
#                        or play-strength metric)
#   value_mae        -- mean absolute error between predicted and target
#                        value, in the same [-1, 1] units as squash_value()
#
# Both train_one_epoch() and evaluate() are dataset-agnostic: they just
# consume a DataLoader yielding (input, policy_idx, value) batches, so the
# same functions work on a tiny 3-game file (this script's demo) or a full
# multi-GB Lichess export.
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from network_model import ChessNet, get_device, amp_context
from pgn_pipeline import PGNIterableDataset, split_train_val


def compute_losses(policy_logits, value_pred, policy_target, value_target):
    """
    policy_logits: (batch, POLICY_OUTPUT_SIZE) raw logits from ChessNet.forward()
    value_pred:    (batch,) tanh-squashed value predictions
    policy_target: (batch,) long tensor of played-move policy indices
    value_target:  (batch,) float tensor of side-to-move-relative values

    Returns (policy_loss, value_loss, total_loss) as scalar tensors.
    Cross-entropy on raw logits against the human-played move index; MSE
    on the value head. Losses are summed (not weighted) -- both are
    already on comparable scales (~O(1)) since policy_loss starts around
    ln(4672)=8.4 and value MSE starts around 0.3-1.0 for a random network,
    which is a reasonable, commonly-used starting point for this
    architecture. Revisit with a weighting term if one head visibly
    dominates training once real-scale data is used.
    """
    policy_loss = nn.functional.cross_entropy(policy_logits, policy_target)
    value_loss = nn.functional.mse_loss(value_pred, value_target)
    total_loss = policy_loss + value_loss
    return policy_loss, value_loss, total_loss


def _run_epoch(model, loader, device, optimizer=None, scaler=None):
    """
    Shared implementation for train_one_epoch() and evaluate(). If
    optimizer is None, runs in no-grad eval mode; otherwise trains.
    """
    is_train = optimizer is not None
    model.train(is_train)

    sum_policy_loss = 0.0
    sum_value_loss = 0.0
    sum_total_loss = 0.0
    sum_correct = 0
    sum_abs_err = 0.0
    num_examples = 0
    num_batches = 0

    grad_context = torch.enable_grad() if is_train else torch.inference_mode()
    with grad_context:
        for input_tensor, policy_target, value_target in loader:
            input_tensor = input_tensor.to(device)
            policy_target = policy_target.to(device)
            value_target = value_target.to(device)
            batch_size = input_tensor.shape[0]

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with amp_context(device):
                policy_logits, value_pred = model(input_tensor)
                policy_loss, value_loss, total_loss = compute_losses(
                    policy_logits, value_pred, policy_target, value_target
                )

            if is_train:
                if scaler is not None:
                    scaler.scale(total_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    optimizer.step()

            with torch.no_grad():
                predicted_move = policy_logits.argmax(dim=-1)
                sum_correct += (predicted_move == policy_target).sum().item()
                sum_abs_err += (value_pred.float() - value_target.float()).abs().sum().item()

            sum_policy_loss += policy_loss.item() * batch_size
            sum_value_loss += value_loss.item() * batch_size
            sum_total_loss += total_loss.item() * batch_size
            num_examples += batch_size
            num_batches += 1

    if num_examples == 0:
        raise ValueError(
            "epoch saw zero examples -- check that the PGN file/split "
            "actually contains eval-annotated positions"
        )

    return {
        "policy_loss": sum_policy_loss / num_examples,
        "value_loss": sum_value_loss / num_examples,
        "total_loss": sum_total_loss / num_examples,
        "policy_top1_acc": sum_correct / num_examples,
        "value_mae": sum_abs_err / num_examples,
        "num_examples": num_examples,
        "num_batches": num_batches,
    }


def train_one_epoch(model, loader, optimizer, device, scaler=None):
    """Runs one training epoch (forward + backward + optimizer step). Returns a metrics dict."""
    return _run_epoch(model, loader, device, optimizer=optimizer, scaler=scaler)


@torch.inference_mode()
def evaluate(model, loader, device):
    """Runs one evaluation pass with no gradient updates. Returns a metrics dict."""
    return _run_epoch(model, loader, device, optimizer=None, scaler=None)


def _fmt(metrics: dict) -> str:
    return (
        f"total={metrics['total_loss']:.4f}  "
        f"policy={metrics['policy_loss']:.4f}  "
        f"value={metrics['value_loss']:.4f}  "
        f"policy_top1={metrics['policy_top1_acc']:.1%}  "
        f"value_mae={metrics['value_mae']:.4f}  "
        f"(n={metrics['num_examples']})"
    )


def fit(
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    checkpoint_path: str | None = None,
    checkpoint_every: int = 1,
):
    """
    Full train/val loop: AdamW + cosine LR schedule, one call to
    train_one_epoch() and evaluate() per epoch, checkpointing every
    `checkpoint_every` epochs if checkpoint_path is given. Returns the
    per-epoch history as a list of {"epoch", "train": {...}, "val": {...}} dicts.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"epoch {epoch:>2}/{epochs}  ({elapsed:5.1f}s, lr={scheduler.get_last_lr()[0]:.2e})")
        print(f"  train  {_fmt(train_metrics)}")
        print(f"  val    {_fmt(val_metrics)}")

        if checkpoint_path is not None and epoch % checkpoint_every == 0:
            model.save_checkpoint(checkpoint_path, epoch=epoch, optimizer_state_dict=optimizer.state_dict())
            print(f"  saved checkpoint -> {checkpoint_path}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

    return history


# ---------------------------------------------------------------------------
# Demo / verification: run the loop end-to-end on first_3_games.pgn and print
# results that can be checked by hand.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    pgn_path = sys.argv[1] if len(sys.argv) > 1 else "first_3_games.pgn"

    torch.manual_seed(0)
    device = get_device()
    print(f"device: {device}")

    # first_3_games.pgn only has 3 games -- split 2 train / 1 val at the
    # game level (never split a single game's positions across train/val).
    train_ds, val_ds = split_train_val(pgn_path, val_fraction=1 / 3, seed=0)
    train_loader = DataLoader(train_ds, batch_size=8)
    val_loader = DataLoader(val_ds, batch_size=8)

    train_count = sum(1 for _ in train_ds)
    val_count = sum(1 for _ in val_ds)
    print(f"train examples: {train_count}   val examples: {val_count}")
    assert train_count > 0 and val_count > 0, "split produced an empty side -- check the file has >= 2 games"

    # A full-size ChessNet (192 filters x 16 res blocks, ~20.6M params) is
    # far more than 3 games of data can meaningfully train, and is slow on
    # CPU. For this correctness demo we intentionally build a much smaller
    # ChessNet (same architecture/classes, fewer filters/blocks) so the
    # loop runs in seconds -- real training uses network_model.py's
    # defaults (NUM_FILTERS=192, NUM_RES_BLOCKS=16).
    model = ChessNet(num_filters=32, num_res_blocks=2, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"demo model: {num_params:,} params (scaled down from the real ~20.6M-param config for a fast CPU demo)")

    # --- sanity check: loss should DECREASE and policy_top1_acc should
    # INCREASE over a few epochs on this tiny, repeatedly-seen train set,
    # since a network can (over)fit 120 examples quickly. This is the
    # verifiable signal that gradients are flowing correctly through both
    # heads and the optimizer step is actually updating the weights.
    print("\n--- training ---")
    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=5,
        lr=1e-3,
        checkpoint_path="demo_checkpoint.pt",
        checkpoint_every=5,
    )

    first_train_loss = history[0]["train"]["total_loss"]
    last_train_loss = history[-1]["train"]["total_loss"]
    first_train_acc = history[0]["train"]["policy_top1_acc"]
    last_train_acc = history[-1]["train"]["policy_top1_acc"]

    print("\n--- verification ---")
    print(f"train total_loss: {first_train_loss:.4f} -> {last_train_loss:.4f}")
    print(f"train policy_top1_acc: {first_train_acc:.1%} -> {last_train_acc:.1%}")
    loss_decreased = last_train_loss < first_train_loss
    acc_increased = last_train_acc >= first_train_acc
    print(f"loss decreased over training: {loss_decreased}")
    print(f"policy top-1 accuracy increased (or held) over training: {acc_increased}")

    # --- checkpoint round-trip check: reload the saved checkpoint into a
    # fresh model and confirm it reproduces identical val-set metrics,
    # proving save_checkpoint/load_checkpoint preserve the trained weights
    # exactly (not just "a" set of weights).
    print("\n--- checkpoint round-trip check ---")
    reloaded = ChessNet.load_checkpoint("demo_checkpoint.pt", device=device)
    val_before = evaluate(model, val_loader, device)
    val_after = evaluate(reloaded, val_loader, device)
    print(f"val metrics from in-memory model : {_fmt(val_before)}")
    print(f"val metrics from reloaded model  : {_fmt(val_after)}")
    metrics_match = (
        abs(val_before["total_loss"] - val_after["total_loss"]) < 1e-6
        and abs(val_before["policy_top1_acc"] - val_after["policy_top1_acc"]) < 1e-9
    )
    print(f"reloaded checkpoint reproduces identical metrics: {metrics_match}")

    print("\n--- summary ---")
    all_ok = loss_decreased and metrics_match
    print(f"loss decreased over 5 epochs on the training split : {'PASS' if loss_decreased else 'FAIL'}")
    print(f"checkpoint save/load round-trip is exact           : {'PASS' if metrics_match else 'FAIL'}")
    print("\nTraining/testing loop verified end-to-end." if all_ok else "\nSomething looks off -- see details above.")