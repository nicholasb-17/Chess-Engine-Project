# Training/testing loop
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
#                        human move played
#   value_mae        -- mean absolute error between predicted and target
#                        value, in the same [-1, 1] units as squash_value()
import itertools
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
    ln(4672)=8.4 and value MSE starts around 0.3-1.0 for a random network.
    """
    policy_loss = nn.functional.cross_entropy(policy_logits, policy_target)
    value_loss = nn.functional.mse_loss(value_pred, value_target)
    total_loss = policy_loss + value_loss
    return policy_loss, value_loss, total_loss


def _run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    scaler=None,
    log_every: int | None = None,
    log_prefix: str = "",
    checkpoint_every_steps: int | None = None,
    checkpoint_path: str | None = None,
    epoch: int | None = None,
    extra_checkpoint_state: dict | None = None,
    val_loader=None,
    val_every_steps: int | None = None,
    val_max_batches: int | None = None,
    track_game_index: bool = False,
):
    """
    Shared implementation for train_one_epoch() and evaluate(). If
    optimizer is None, runs in no-grad eval mode; otherwise trains.

    log_every: if set, prints a running-metrics line every `log_every` batches.
    checkpoint_every_steps / checkpoint_path: if both are set (train mode
        only), saves a mid-epoch checkpoint every `checkpoint_every_steps`
        batches. This protects a long epoch against losing all progress
        if the runtime disconnects and doesn't interfere with epoch checkpoint saving.
    extra_checkpoint_state: extra key/value pairs (e.g. scheduler_state_dict)
        merged into every mid-epoch checkpoint save, so a resume can restore
        more than just the optimizer.
    val_loader / val_every_steps / val_max_batches: train mode only. If
        val_loader and val_every_steps are both set, runs a *partial*
        validation pass (model.eval() + torch.inference_mode(), via
        evaluate()) every `val_every_steps` training batches, printed
        alongside the train progress line. Full, unbiased val numbers still come from the
        end-of-epoch evaluate() call in fit().
    track_game_index: if True, each batch from `loader` is expected to be
        a 4-tuple (input, policy_idx, value, game_index) -- i.e. `loader`
        wraps a PGNIterableDataset built with yield_game_index=True. The
        highest game_index seen so far is tracked and included as
        `game_index` in every mid-epoch checkpoint (checkpoint_every_steps),
        so a resumed run can fast-forward past already-trained games
        (via PGNIterableDataset's resume_game_index) instead of
        re-streaming the whole epoch from game 0. Train mode only --
        ignored when optimizer is None (evaluate() never needs it).
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

    last_game_index = None  # highest game_index fully seen so far (train + track_game_index only)

    grad_context = torch.enable_grad() if is_train else torch.inference_mode()
    with grad_context:
        for batch in loader:
            if is_train and track_game_index:
                input_tensor, policy_target, value_target, batch_game_index = batch
                # game_index is monotonically increasing within a single
                # worker's file scan; the max across the batch is the
                # furthest we've now consumed a full batch from.
                last_game_index = int(batch_game_index.max().item())
            else:
                input_tensor, policy_target, value_target = batch
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

            if log_every and num_batches % log_every == 0:
                print(
                    f"{log_prefix}batch {num_batches:>6}  n={num_examples:>10,}  "
                    f"total={sum_total_loss / num_examples:.4f}  "
                    f"policy={sum_policy_loss / num_examples:.4f}  "
                    f"value={sum_value_loss / num_examples:.4f}  "
                    f"policy_top1={sum_correct / num_examples:.1%}"
                )

            if is_train and checkpoint_every_steps and checkpoint_path and num_batches % checkpoint_every_steps == 0:
                checkpoint_extra = dict(extra_checkpoint_state or {})
                if track_game_index and last_game_index is not None:
                    checkpoint_extra["game_index"] = last_game_index
                model.save_checkpoint(
                    checkpoint_path,
                    epoch=epoch,
                    step=num_batches,
                    mid_epoch=True,
                    optimizer_state_dict=optimizer.state_dict(),
                    **checkpoint_extra,
                )
                game_index_note = f"  game_index={last_game_index}" if track_game_index else ""
                print(f"{log_prefix}  saved mid-epoch checkpoint at batch {num_batches} -> {checkpoint_path}{game_index_note}")

            if is_train and val_every_steps and val_loader is not None and num_batches % val_every_steps == 0:
                val_iter = (
                    val_loader if val_max_batches is None else itertools.islice(val_loader, val_max_batches)
                )
                mid_val_metrics = evaluate(model, val_iter, device)
                print(f"{log_prefix}  mid-epoch val (batch {num_batches})  {_fmt(mid_val_metrics)}")
                model.train(True)  # evaluate() left the model in eval mode -- restore for training to resume

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


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    log_every: int | None = None,
    log_prefix: str = "",
    checkpoint_every_steps: int | None = None,
    checkpoint_path: str | None = None,
    epoch: int | None = None,
    extra_checkpoint_state: dict | None = None,
    val_loader=None,
    val_every_steps: int | None = None,
    val_max_batches: int | None = None,
    track_game_index: bool = False,
):
    """
    Runs one training epoch (forward + backward + optimizer step). Returns
    a metrics dict.

    val_loader / val_every_steps / val_max_batches: optional periodic
    mid-epoch validation -- see _run_epoch()'s docstring for details.
    track_game_index: see _run_epoch()'s docstring.
    """
    return _run_epoch(
        model,
        loader,
        device,
        optimizer=optimizer,
        scaler=scaler,
        log_every=log_every,
        log_prefix=log_prefix,
        checkpoint_every_steps=checkpoint_every_steps,
        checkpoint_path=checkpoint_path,
        epoch=epoch,
        extra_checkpoint_state=extra_checkpoint_state,
        val_loader=val_loader,
        val_every_steps=val_every_steps,
        val_max_batches=val_max_batches,
        track_game_index=track_game_index,
    )


@torch.inference_mode()
def evaluate(model, loader, device, log_every: int | None = None, log_prefix: str = ""):
    """Runs one evaluation pass with no gradient updates. Returns a metrics dict."""
    return _run_epoch(model, loader, device, optimizer=None, scaler=None, log_every=log_every, log_prefix=log_prefix)


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
    log_every: int | None = None,
    checkpoint_every_steps: int | None = None,
    resume_from: str | None = None,
    val_every_steps: int | None = None,
    val_max_batches: int | None = None,
    track_game_index: bool = False,
):
    """
    Full train/val loop: AdamW + cosine LR schedule, one call to
    train_one_epoch() and evaluate() per epoch, checkpointing every
    `checkpoint_every` epochs if checkpoint_path is given. Returns the
    per-epoch history as a list of {"epoch", "train": {...}, "val": {...}} dicts.

    log_every: if set, prints running train/val metrics every `log_every`
        batches within each epoch
    val_every_steps / val_max_batches: if val_every_steps is set, also runs
        a quick partial validation pass (model.eval() + torch.inference_mode())
        every `val_every_steps` training batches, printed alongside the
        train progress line
    checkpoint_every_steps: if set (and checkpoint_path is given), also
        saves a checkpoint every `checkpoint_every_steps` batches *during*
        training, not just at epoch boundaries.
    resume_from: path to a checkpoint saved by this function (or by
        `model.save_checkpoint()` mid-epoch, via checkpoint_every_steps
        above) to resume from. Restores optimizer and LR-scheduler state
        so training continues the cosine schedule from where it left off,
        rather than restarting it at step 0.
    track_game_index: if True, mid-epoch checkpoints (see
        checkpoint_every_steps) record the furthest game_index reached so
        far, enabling the fast-forward resume described above. Requires
        train_loader to wrap a PGNIterableDataset built with
        yield_game_index=True, or batches won't have a game_index to
        report. Has no effect on epoch-boundary checkpoints' correctness,
        only on whether a resume from a mid-epoch checkpoint can skip
        ahead.
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 1
    if resume_from is not None:
        checkpoint = torch.load(resume_from, map_location=device)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            print(
                f"  warning: {resume_from} has no scheduler_state_dict "
                "(older checkpoint?) -- LR schedule resumes at its initial position"
            )
        saved_epoch = checkpoint.get("epoch", 0)
        saved_game_index = checkpoint.get("game_index")
        # A mid-epoch checkpoint's epoch was still in progress -- re-run it
        # from the start rather than skipping to the next one, since the
        # data pipeline can't resume partway through the file anyway
        # (unless the checkpoint has a game_index -- see below).
        start_epoch = saved_epoch if checkpoint.get("mid_epoch") else saved_epoch + 1
        print(
            f"  resumed from {resume_from}: continuing at epoch {start_epoch}/{epochs} "
            f"(lr={scheduler.get_last_lr()[0]:.2e})"
        )
        if checkpoint.get("mid_epoch") and saved_game_index is not None:
            print(
                f"  checkpoint has game_index={saved_game_index} -- to skip already-trained "
                f"games this epoch, rebuild train_loader's dataset with "
                f"resume_game_index={saved_game_index + 1}, yield_game_index=True, and pass "
                "track_game_index=True to this fit() call"
            )
        elif checkpoint.get("mid_epoch"):
            print(
                "  no game_index in this checkpoint (saved without track_game_index=True) -- "
                "this epoch will re-run from game 0"
            )
        if start_epoch > epochs:
            print(
                f"  warning: resumed epoch {start_epoch} is already past epochs={epochs} -- "
                "nothing left to train; pass a larger epochs value to continue"
            )

    history = []
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            log_every=log_every,
            log_prefix="  train  ",
            checkpoint_every_steps=checkpoint_every_steps,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            extra_checkpoint_state={"scheduler_state_dict": scheduler.state_dict()},
            val_loader=val_loader,
            val_every_steps=val_every_steps,
            val_max_batches=val_max_batches,
            track_game_index=track_game_index,
        )
        val_metrics = evaluate(model, val_loader, device, log_every=log_every, log_prefix="  val    ")
        scheduler.step()
        elapsed = time.time() - t0

        print(f"epoch {epoch:>2}/{epochs}  ({elapsed:5.1f}s, lr={scheduler.get_last_lr()[0]:.2e})")
        print(f"  train  {_fmt(train_metrics)}")
        print(f"  val    {_fmt(val_metrics)}")

        if checkpoint_path is not None and epoch % checkpoint_every == 0:
            model.save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                mid_epoch=False,
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict(),
            )
            print(f"  saved checkpoint -> {checkpoint_path}")

        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

    return history


# ---------------------------------------------------------------------------
# Demo / verification: run the loop end-to-end on first_3_games.pgn (self made file for testing purposes and not include in github repo) and print results.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    #defaults to first_3_games.pgn if no argument is provided, which is a small PGN file with 3 games for testing purposes.
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

    # build a much smaller ChessNet (same architecture/classes, fewer filters/blocks)
    # loop runs in seconds -- real training uses network_model.py's
    # defaults (NUM_FILTERS=192, NUM_RES_BLOCKS=16).
    model = ChessNet(num_filters=32, num_res_blocks=2, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"demo model: {num_params:,} params (scaled down from the real ~20.6M-param config for a fast CPU demo)")

    # --- sanity check: loss should DECREASE and policy_top1_acc should
    # INCREASE over a few epochs on this tiny, repeatedly-seen train set,
    # since a network can (over)fit 120 examples quickly.
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