# !pip install chess for running in colab, otherwise it won't work
import os
import sys
import torch
from torch.utils.data import DataLoader

from google.colab import drive
drive.mount('/content/drive')

# Ensure drive path is added before importing local modules
sys.path.append('/content/drive/MyDrive/Chess_Engine_Files')

#was bugging before so added this to make sure it is in the path
#!mkdir -p /content/chess_engine
#!cp /content/drive/MyDrive/Chess_Engine_Files/*.py /content/chess_engine/
#sys.path.insert(0, '/content/chess_engine')

from network_model import ChessNet, get_device
from pgn_pipeline import split_train_val, PGNIterableDataset
from train_and_test_loop import fit
device = get_device()

pgn_path = '/content/drive/MyDrive/Chess_Engine_Files/filtered_games.pgn'
checkpoint_path = '/content/drive/MyDrive/Chess_Engine_Files/checkpoints/model.pt'

# 1. Load checkpoint if present; otherwise initialize a fresh model
if os.path.exists(checkpoint_path):
    print(f"Loading existing checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    resume_idx = ckpt.get("game_index")
    model = ChessNet.load_checkpoint(checkpoint_path, device=device)
    resume_from_path = checkpoint_path
else:
    print("No checkpoint found. Initializing a fresh ChessNet model.")
    ckpt = None
    resume_idx = None
    model = ChessNet().to(device)
    resume_from_path = None

# Ensure the checkpoint directory exists before training so saving won't fail later
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

# 2. Split dataset indices
train_indices, val_indices = split_train_val(pgn_path)

# 3. Instantiate datasets and loaders
train_ds = PGNIterableDataset(
    pgn_path,
    game_indices=train_indices,
    resume_game_index=(resume_idx + 1) if resume_idx is not None else 0,
    yield_game_index=True,
)
val_ds = PGNIterableDataset(
    pgn_path,
    game_indices=val_indices,
    yield_game_index=False,
)

train_loader = DataLoader(train_ds, batch_size=256)
val_loader = DataLoader(val_ds, batch_size=256)

# 4. Train
history = fit(
    model,
    train_loader,
    val_loader,
    epochs=3,
    checkpoint_path=checkpoint_path,
    checkpoint_every=1,
    log_every=200,
    checkpoint_every_steps=2000,
    resume_from=resume_from_path,
    track_game_index=True,
)