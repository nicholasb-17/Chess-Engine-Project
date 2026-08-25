# PGN data pipeline
# Turns a Lichess PGN file (with embedded [%eval ...] annotations, as
# produced by Lichess's "Rated games" export with evals enabled) into
# training examples for network_model.py's ChessNet:
#   input  : board_to_tensor(board)                     (105, 8, 8)
#   policy : encode_move(board, played_move)             int in [0, 4672)
#   value  : side-to-move-relative eval, squashed to [-1, 1]
# Streams the file game-by-game with chess.pgn.read_game() rather than
# loading everything into memory.
# Dataset that needs __len__/__getitem__ random access.
import math
import re
import chess
import chess.pgn
import torch
from torch.utils.data import IterableDataset

from dataset import board_to_tensor
from engine_output_dataset import encode_move

# Lichess eval comments look like "[%eval 0.18]" (pawns, White POV) or
# "[%eval #-3]" (forced mate in 3 for Black, White POV sign convention).
_EVAL_RE = re.compile(r"\[%eval\s+(#?-?\d+(?:\.\d+)?)\]")

# Mate scores get mapped to a centipawn value large enough that
# tanh(cp / EVAL_SQUASH_DIVISOR) saturates to (near) +/-1 regardless of
# how many moves the mate is in -- a mate-in-1 and a mate-in-8 are both
# just "someone is winning decisively" for value-head purposes.
MATE_SCORE_CP = 10000.0
EVAL_SQUASH_DIVISOR = 400.0  # tanh(cp / 400): +/-400cp (a rook) -> ~+/-0.76


def parse_eval_cp(comment: str) -> float | None:
    """
    Extract the White-POV evaluation from a move comment, in centipawns.
    Returns None if the comment has no [%eval ...] tag (e.g. the final
    mating move, or a PGN exported without eval annotations at all).
    Mate scores ("#3", "#-3") are mapped to +/-MATE_SCORE_CP.
    """
    match = _EVAL_RE.search(comment)
    if match is None:
        return None
    raw = match.group(1)
    if raw.startswith("#"):
        mate_in = int(raw[1:])
        return math.copysign(MATE_SCORE_CP, mate_in)
    return float(raw) * 100.0  # pawns -> centipawns


def squash_value(eval_cp: float, mover_is_white: bool) -> float:
    """
    Convert a White-POV centipawn eval into a side-to-move-relative value
    in [-1, 1] via tanh, flipping sign when the side to move (the player
    about to make the move this example is labeling) is Black.
    """
    value = math.tanh(eval_cp / EVAL_SQUASH_DIVISOR)
    return value if mover_is_white else -value


def generate_examples_from_game(game: chess.pgn.Game):
    """
    Walk a parsed game's mainline and yield (board, played_move, value)
    tuples, where `board` is the position BEFORE played_move (oriented
    with real, unmirrored coordinates -- board_to_tensor/encode_move do
    their own orientation), and `value` is in [-1, 1] relative to
    whoever is to move on `board`. `board` is a fresh, independent chess.Board per example (not a
    shared mutable reference)
    """
    board = game.board()
    node = game
    while node.variations:
        next_node = node.variations[0]
        move = next_node.move
        eval_cp = parse_eval_cp(next_node.comment or "")
        if eval_cp is not None:
            mover_is_white = board.turn == chess.WHITE
            value = squash_value(eval_cp, mover_is_white)
            yield board.copy(stack=True), move, value
        board.push(move)
        node = next_node


class PGNIterableDataset(IterableDataset):
    """
    Streams a PGN file into (input_tensor, policy_index, value_tensor)
    training triples.

    min_white_elo / min_black_elo: optional quality filters (games below
        either threshold are skipped entirely). Set to None to disable.
    skip_no_eval_games: if True, games with zero eval-annotated positions
        are skipped outright (cheap early-out before wasting board_to_tensor
        calls on a game you can't get value targets from).
    game_indices: optional set/collection of 0-based game indices (in file
        order) to restrict this dataset to. None (default) means "every
        game in the file". Used to carve a single PGN file into disjoint
        train/val splits at the game level (never split a single game's
        positions across train and val -- that would leak near-duplicate,
        highly correlated positions across the split).
    resume_game_index: 0-based game index (in file order) to fast-forward
        to before yielding anything. Games before this index are skipped
        with chess.pgn.skip_game(), which only scans past the game's PGN
        text. Default 0 means"start from the top of the file", identical to prior behavior.
    yield_game_index: if True, each yielded example is a 4-tuple
        (input_tensor, policy_index, value_tensor, game_index) instead of
        the usual 3-tuple, so a training loop can track how far into the
        file it's gotten and record that in checkpoints (see
        train_and_test_loop.py's track_game_index). Default False keeps
        the original 3-tuple shape for any existing code that unpacks it.
    """

    def __init__(
        self,
        pgn_path: str,
        min_white_elo: int | None = None,
        min_black_elo: int | None = None,
        skip_no_eval_games: bool = True,
        game_indices: "set[int] | None" = None,
        resume_game_index: int = 0,
        yield_game_index: bool = False,
    ):
        super().__init__()
        self.pgn_path = pgn_path
        self.min_white_elo = min_white_elo
        self.min_black_elo = min_black_elo
        self.skip_no_eval_games = skip_no_eval_games
        self.game_indices = game_indices
        self.resume_game_index = resume_game_index
        self.yield_game_index = yield_game_index

    def _game_passes_filters(self, game: chess.pgn.Game) -> bool:
        headers = game.headers
        if self.min_white_elo is not None:
            try:
                if int(headers.get("WhiteElo", "0")) < self.min_white_elo:
                    return False
            except ValueError:
                return False
        if self.min_black_elo is not None:
            try:
                if int(headers.get("BlackElo", "0")) < self.min_black_elo:
                    return False
            except ValueError:
                return False
        return True

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        with open(self.pgn_path, encoding="utf-8", errors="replace") as pgn_file:
            game_index = 0

            # Fast-forward past already-consumed games without parsing
            # moves into tensors (or even building a Game object) --
            # see resume_game_index in the class docstring.
            while game_index < self.resume_game_index:
                if chess.pgn.skip_game(pgn_file) is False:
                    return  # file ended before reaching resume_game_index
                game_index += 1

            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break  # end of file

                this_index = game_index
                game_index += 1
                if this_index % num_workers != worker_id:
                    continue  # another worker owns this game

                if self.game_indices is not None and this_index not in self.game_indices:
                    continue  # not part of this split

                if not self._game_passes_filters(game):
                    continue

                for board, move, value in generate_examples_from_game(game):
                    input_tensor = board_to_tensor(board)
                    policy_index = encode_move(board, move)
                    value_tensor = torch.tensor(value, dtype=torch.float32)
                    policy_tensor = torch.tensor(policy_index, dtype=torch.long)
                    if self.yield_game_index:
                        yield input_tensor, policy_tensor, value_tensor, this_index
                    else:
                        yield input_tensor, policy_tensor, value_tensor


def count_games(pgn_path: str) -> int:
    """
    Count total games in a PGN file by scanning headers only .
    """
    count = 0
    with open(pgn_path, encoding="utf-8", errors="replace") as pgn_file:
        while chess.pgn.read_headers(pgn_file) is not None:
            count += 1
    return count


def split_train_val(
    pgn_path: str, val_fraction: float = 0.1, seed: int = 0, **dataset_kwargs
) -> tuple["PGNIterableDataset", "PGNIterableDataset"]:
    """
    Split a PGN file into train/val PGNIterableDatasets at the game level
    (not the position level), so no game's positions appear in both splits.
    Returns (train_dataset, val_dataset). Extra dataset_kwargs (elo filters,
    skip_no_eval_games) are forwarded to both.
    """
    import random

    total_games = count_games(pgn_path)
    rng = random.Random(seed)
    all_indices = list(range(total_games))
    rng.shuffle(all_indices)
    num_val = max(1, int(total_games * val_fraction)) if total_games > 1 else 0
    val_indices = set(all_indices[:num_val])
    train_indices = set(all_indices[num_val:])
    train_ds = PGNIterableDataset(pgn_path, game_indices=train_indices, **dataset_kwargs)
    val_ds = PGNIterableDataset(pgn_path, game_indices=val_indices, **dataset_kwargs)
    return train_ds, val_ds


# sanity checks / demo
if __name__ == "__main__":
    import sys

    pgn_path = sys.argv[1] if len(sys.argv) > 1 else "first_3_games.pgn"

    # 1. eval parsing sanity checks
    assert parse_eval_cp("[%eval 0.18] [%clk 0:05:00]") == 18.0
    assert parse_eval_cp("[%eval -1.38] [%clk 0:04:35]") == -138.0
    assert parse_eval_cp("[%clk 0:01:12]") is None  # final mating move, no eval
    mate_for_white = parse_eval_cp("[%eval #3] [%clk 0:03:18]")
    mate_for_black = parse_eval_cp("[%eval #-3] [%clk 0:03:18]")
    assert mate_for_white == MATE_SCORE_CP
    assert mate_for_black == -MATE_SCORE_CP
    print("parse_eval_cp: all checks passed")

    # 2. squash_value sanity checks
    v_white = squash_value(400.0, mover_is_white=True)
    v_black = squash_value(400.0, mover_is_white=False)
    assert math.isclose(v_white, math.tanh(1.0))
    assert math.isclose(v_black, -math.tanh(1.0))
    assert -1.0 <= v_white <= 1.0 and -1.0 <= v_black <= 1.0
    print("squash_value: all checks passed")

    # 3. walk the first game directly with generate_examples_from_game
    with open(pgn_path, encoding="utf-8") as f:
        first_game = chess.pgn.read_game(f)
    examples = list(generate_examples_from_game(first_game))
    print(f"\nfirst game: {len(examples)} eval-labeled examples")
    for board, move, value in examples[:3]:
        print(f"  ply {len(board.move_stack):>2}  {board.san(move):<8} value={value:+.3f}  mover={'White' if board.turn else 'Black'}")

    # 4. end-to-end through the IterableDataset (tensor shapes match ChessNet's expectations)
    ds = PGNIterableDataset(pgn_path)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=8)
    batch = next(iter(loader))
    x, policy_idx, value = batch
    print(f"\nfirst batch: input={tuple(x.shape)} policy_idx={tuple(policy_idx.shape)} value={tuple(value.shape)}")
    assert x.shape[1:] == (105, 8, 8)
    assert policy_idx.dtype == torch.long
    assert value.dtype == torch.float32
    assert (value.abs() <= 1.0).all()
    print("IterableDataset: shapes and dtypes match ChessNet's forward()/loss expectations")

    # 5. count total examples across all 3 games, as a smoke test over the whole file
    total = sum(1 for _ in PGNIterableDataset(pgn_path))
    print(f"\ntotal eval-labeled examples across file: {total}")
    print("\nall pgn_pipeline sanity checks passed")