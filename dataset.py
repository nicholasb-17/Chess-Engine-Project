import chess
import numpy as np
import torch

#board.push_san("e4")
#board.push_san("e5")
#board.push_san("Qh5")
#board.push_san("Nc6")
#board.push_san("Bc4")
#board.push_san("Nf6")
#board.push_san("Qxf7")

#python -c "import myscript; myscript.my_function()"
#def function(parameter: class) -> return_type:

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Converts a python-chess Board (object class built using bitboards) into an 21x8x8 numerical tensor (channel, height, width).
    Channels 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    Channels 6-11: Black pieces (same order)
    Channel 12: side to move (all 1s if White to move, all 0s if Black)
    Channels 13-14: White kingside / queenside castling rights (all 1s or all 0s)
    Channels 15-16: Black kingside / queenside castling rights (all 1s or all 0s)
    Channel 17: en passant target square (1 at that square, 0 elsewhere)
    Channel 18: side to move is in check (all 1s or all 0s)
    Channel 19: side to move is checkmated (all 1s or all 0s)
    Channel 20: side to move is stalemated (all 1s or all 0s)
    """
#   Creating an empty tensor
    tensor = np.zeros((21, 8, 8), dtype=np.float32)
#   Mapping the board position to a dictionnary with keys from 0-63 (squares) and values as chess. Piece are objects with attributes .piece_type (from 1 to 6/ from pawn to king) and .color (True = White, False = Black)
    piece_map = board.piece_map()
#   Filling the tensor with piece positions by iterating over the piece_map dictionary
    for square, piece in piece_map.items():
#       Getting the rank (rows) and file (columns) indices from the square index (0-63)
        rank, file = divmod(square, 8) 
#       Reindexing to match tensor representation
        piece_type = piece.piece_type - 1
#       Mapping the piece type and color to the appropriate channel in the tensor (0-11 for White pieces, 6-11 for Black pieces)
        offset = 0 if piece.color == chess.WHITE else 6
        tensor[piece_type + offset, rank, file] = 1.0
    
#   Side to move: fill the whole channel-12 plane with 1.0 if it's White's turn, else 0.0
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

#   Castling rights: each is a global (thus, not square-specific) fact, so the whole plane is filled
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

#   En passant target square: mark the single square where an en passant capture is possible, if any
    if board.ep_square is not None:
        ep_rank, ep_file = divmod(board.ep_square, 8)
        tensor[17, ep_rank, ep_file] = 1.0

#   Check / checkmate / stalemate: global facts about the side to move, so fill whole planes
    tensor[18, :, :] = 1.0 if board.is_check() else 0.0
    tensor[19, :, :] = 1.0 if board.is_checkmate() else 0.0
    tensor[20, :, :] = 1.0 if board.is_stalemate() else 0.0

    return torch.from_numpy(tensor)

#need to add:
#board orientation
#8 move history
#is_game_over() (for checkmate, stalemate, insufficient material, 50-move rule, threefold repetition)
#halfmove clock and fullmove number (for 50-move rule and game history) (normalized)
#repitition count
