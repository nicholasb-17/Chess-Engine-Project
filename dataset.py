# chess.py is a Python chess library that provides a simple and efficient way to represent and manipulate chess positions, moves, and games.
# Numpy is a powerful library for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices.
# Pytorch is an open-source machine learning library.
import chess
import numpy as np
import torch

# Tunable variables
HISTORY_LENGTH = 7          # number of past positions to stack. 8 total positions (current + 7 history) are encoded in the tensor.
HALFMOVE_CLOCK_CAP = 100.0  # the 50-move rule (for draws) triggers at 100 half-moves
FULLMOVE_CAP = 500.0        # arbitrary large-game cap, just for normalization


def _orient(board: chess.Board, white_to_move: bool) -> chess.Board:
    """
    Return `board` reoriented so it is always viewed from the perspective of
    whoever is currently on move. `python-chess`'s .mirror() flips the board
    vertically (rank 0 <-> rank 7), swaps piece colors, and re-derives castling rights and en passant squares.
    After mirroring, the mover's own pieces are always chess.WHITE.

    """
    return board if white_to_move else board.mirror()


def _piece_planes(oriented_board: chess.Board) -> np.ndarray:
    """
    Encode the 12 piece-placement planes for a board that has already been
    oriented so that the side to move is chess.WHITE.
    Channels 0-5: side-to-move's pieces (pawn to king)
    Channels 6-11: opponent's pieces (pawn to king)

    """
    #empty tensor of shape (12, 8, 8) to hold the piece planes
    planes = np.zeros((12, 8, 8), dtype=np.float32)
    #iterate over the piece map of the oriented board and fill in the planes
    for square, piece in oriented_board.piece_map().items():
        rank, file = divmod(square, 8)
        piece_type = piece.piece_type - 1
        offset = 0 if piece.color == chess.WHITE else 6
        planes[piece_type + offset, rank, file] = 1.0
    return planes

def _repetition_count(board: chess.Board) -> int:
    """
    How many times the current position has already occurred in the game including the current position, up to three times
    by scanning the last irreversible move.

    """
    # uses `python-chess`'s built-in repetition detection, which counts the number of times the current position has occurred in the game.
    if board.is_repetition(3):
        return 3
    if board.is_repetition(2):
        return 2
    return 1


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a python-chess Board into a (12 + 7*12 + 1 + 4 + 1 + 1 + 1 + 1 + 1)
    = 106x 8 x 8 tensor (channel, height, width).

    Channels  0-11  : The current position; oriented to the side to move
                       (0-5 = side to move's pieces, 6-11 = opponent's pieces).
    Channels 12-95  : 7-step move history that are 12 planes each, with the oldest indicated earlier,
                       and oriented consistently with the current position with missing history being zero-padded.
    Channel     96  : side to move (all 1s if White, all 0s if Black)
    Channel     97  : side to move's kingside castling rights
    Channel     98  : side to move's queenside castling rights
    Channel     99  : opponent's kingside castling rights
    Channel     100 : opponent's queenside castling rights
    Channel     101 : en passant target square (oriented coordinates)
    Channel     102 : side to move is in check
    Channel     103 : halfmove clock, normalized to [0, 1] (50-move rule)
    Channel     104 : fullmove number, normalized to [0, 1] (capped)
    Channel     105 : repetition count --> normalized to [0.0,0.5,1.0] for 1, 2, or 3 repetitions respectively

    """
    white_to_move = board.turn == chess.WHITE
    channels = []

    # current position
    oriented_current = _orient(board, white_to_move)
    channels.append(_piece_planes(oriented_current))

    # 7-step history
    # Walk backward through the move stack without mutating the caller's board.
    temp = board.copy(stack=True)
    for i in range(HISTORY_LENGTH):
        if temp.move_stack:
            temp.pop()
            oriented_past = _orient(temp, white_to_move)
            channels.append(_piece_planes(oriented_past))
        else:
            # No more history available (near the start of the game) -> pad with zeros.
            channels.append(np.zeros((12, 8, 8), dtype=np.float32))

    # side to move
    channels.append(np.full((1, 8, 8), 1.0 if white_to_move else 0.0, dtype=np.float32))

    # castling rights, relative to the mover
    own_color, opp_color = board.turn, not board.turn
    channels.append(np.full((1, 8, 8), float(board.has_kingside_castling_rights(own_color)), dtype=np.float32))
    channels.append(np.full((1, 8, 8), float(board.has_queenside_castling_rights(own_color)), dtype=np.float32))
    channels.append(np.full((1, 8, 8), float(board.has_kingside_castling_rights(opp_color)), dtype=np.float32))
    channels.append(np.full((1, 8, 8), float(board.has_queenside_castling_rights(opp_color)), dtype=np.float32))

    # en passant target square in oriented coordinates
    ep_plane = np.zeros((1, 8, 8), dtype=np.float32)
    if oriented_current.ep_square is not None:
        ep_rank, ep_file = divmod(oriented_current.ep_square, 8)
        ep_plane[0, ep_rank, ep_file] = 1.0
    channels.append(ep_plane)

    # if position is in check
    channels.append(np.full((1, 8, 8), 1.0 if board.is_check() else 0.0, dtype=np.float32))

    # normalized halfmove clock
    hc = min(board.halfmove_clock, HALFMOVE_CLOCK_CAP) / HALFMOVE_CLOCK_CAP
    channels.append(np.full((1, 8, 8), hc, dtype=np.float32))

    # normalized fullmove number
    fm = min(board.fullmove_number, FULLMOVE_CAP) / FULLMOVE_CAP
    channels.append(np.full((1, 8, 8), fm, dtype=np.float32))

    # normalized repetition count (0.0 for 1 occurrence, 0.5 for 2 occurrences, 1.0 for 3 occurrences)
    rep = (_repetition_count(board) - 1) / 2.0
    channels.append(np.full((1, 8, 8), rep, dtype=np.float32))

    #creating tensor along channel axis
    tensor = np.concatenate(channels, axis=0)
    return torch.from_numpy(tensor)

#sanity checks
if __name__ == "__main__":
    board = chess.Board()
    #using SAN (Standard Algebraic Notation) to make moves
    for move in ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7"]:
        board.push_san(move)
    amongus = board_to_tensor(board)
    print("shape:", amongus.shape)
    assert amongus.shape == (106, 8, 8)
    print("side to move plane (should be 0s, Black to move):", amongus[96, 0, 0].item())
    print("check plane (should be 1, Qxf7 is check):", amongus[102, 0, 0].item())
    print("halfmove clock plane:", amongus[103, 0, 0].item())
    print("fullmove number plane:", amongus[104, 0, 0].item())

    
    # sanity check on 1 move game
    board2 = chess.Board()
    board2.push_san("e4")
    amongus2 = board_to_tensor(board2)
    # frame at channel 12 (1 ply back) should be the empty starting-ish position for that ply,
    # frames further back (no history) should be all zero
    oldest_frame_start = 12 + 6 * 12
    print("oldest history frame is all zero (no history at move 1):", (amongus2[oldest_frame_start:oldest_frame_start + 12] == 0).all().item(),)

    # sanity check for threefold repetition detection
    board3 = chess.Board()
    for move in ["Nf3", "Nf6", "Ng1", "Ng8", "Nf3", "Nf6", "Ng1", "Ng8"]:
        board3.push_san(move)
    amongus3 = board_to_tensor(board3)
    print("repetition plane (should be 1.0, threefold reached):", amongus3[105, 0, 0].item())