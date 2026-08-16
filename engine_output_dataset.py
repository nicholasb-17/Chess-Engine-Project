# Move <-> policy-index encoding, matching the 8x8x73 AlphaZero-style used by network_model.py's PolicyHead (POLICY_OUTPUT_SIZE = 8*8*73 = 4672).
# Move Encoding layout:
#   planes  0-55 : "queen-like" moves - 8 directions x 7 distances
#                  (covers rook/bishop/queen slides, king single steps,
#                  pawn single/double forward pushes, pawn diagonal captures,
#                  and queen promotions)
#   planes 56-63 : knight moves (8 possible knight deltas)
#   planes 64-72 : underpromotions - 3 directions (straight, capture-left,
#                  capture-right) x 3 promotion pieces (knight, bishop, rook)
# Every move is encoded relative to the ORIENTED board (i.e. from the
# perspective of whoever's turn it is to move, matching dataset.py's `_orient()``), so a
# move by Black is first mirrored the same way dataset.py mirrors the board,
# encoded, and un-mirrored on the way back out. This keeps the policy output
# consistent regardless of which side is moving.
import chess

# Global Constants
NUM_OF_SQUARES = 64
PLANES_PER_SQUARE = 73
POLICY_OUTPUT_SIZE = NUM_OF_SQUARES * PLANES_PER_SQUARE  # 4672, to match network_model.py
# 8 directional vectors in oriented (file, rank) space: N, NE, E, SE, S, SW, W, NW (including pawn-queen promotion)
QUEEN_DIRECTIONS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
# The 8 possible knight vectors(delta_file, delta_rank) knight jumps, fixed arbitrary order
KNIGHT_DELTAS = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
# Underpromotion direction vectors, in oriented space (pawn always moves toward +rank)
UNDERPROMO_DIRECTIONS = [0, -1, 1]  # straight, capture-left, capture-right
UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _orient_move(move: chess.Move, white_to_move: bool) -> chess.Move:
    """
    Reorient a move the same way dataset.py's `_orient()` reorients a board:
    if it's Black to move, mirror the from/to squares vertically (rank flip)
    so the move is expressed from the mover's own perspective.
    """
    if white_to_move:
        return move
    #black to move, mirror the from/to squares vertically (rank flip)
    #promotion defaults to none if ommited
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)


def encode_move(board: chess.Board, move: chess.Move) -> int:
    """
    Convert a legal move on `board` (in the board's real, unoriented
    coordinates) into a policy index in [0, POLICY_OUTPUT_SIZE).
    """
    white_to_move = board.turn == chess.WHITE
    oriented = _orient_move(move, white_to_move)
    # piece move delta calculation in oriented coordinates (file, rank)
    from_rank, from_file = divmod(oriented.from_square, 8)
    to_rank, to_file = divmod(oriented.to_square, 8)
    df = to_file - from_file
    dr = to_rank - from_rank

    #check for underpromotion
    if oriented.promotion is not None and oriented.promotion != chess.QUEEN:
        # underpromotion: knight, bishop, or rook
        direction_index = UNDERPROMO_DIRECTIONS.index(df)
        piece_index = UNDERPROMO_PIECES.index(oriented.promotion)
        plane = 64 + direction_index * 3 + piece_index
    #check for knight move
    elif (df, dr) in KNIGHT_DELTAS:
        plane = 56 + KNIGHT_DELTAS.index((df, dr))
    #check for queen or queen-like move (rook, bishop, king, pawn push/capture)
    else:
        # queen-like move (also covers queen promotions, ie: a queen promotion and a same-direction/same-distance slide by an already existing queen/rook/bishop/king/pawn)
        #This means different moves can encode to the same plane.
        #This is fine on the encode side since only one piece can occupy the from-square, but it means decode_index() must consult board state to disambiguate.
        #checking sign of df and dr to find the direction index in QUEEN_DIRECTIONS
        sign = lambda v: (v > 0) - (v < 0)
        direction_index = QUEEN_DIRECTIONS.index((sign(df), sign(dr)))
        distance = max(abs(df), abs(dr))
        #only seven max squares to move to, not 8
        plane = direction_index * 7 + (distance - 1)

    #oriented.from_square * PLANES_PER_SQUARE + plane gives the final policy index as an integer in the range [0, 4671], with oriented.from_square in [0, 63]
    return oriented.from_square * PLANES_PER_SQUARE + plane


def decode_index(board: chess.Board, index: int) -> chess.Move | None:
    """
    Convert a policy index back into a chess.Move in the board's real,
    unoriented coordinates. Returns None if the index doesn't correspond
    to a geometrically valid move (e.g. would move off the board) --
    callers should always verify the result is in board.legal_moves,
    since this function does not know about check legality, only
    geometry plus which piece actually occupies the from-square.

    NOTE on promotions: a queen-like move landing on the back rank (plane
    < 56, to_rank == 7 in oriented coordinates) is only a queen promotion
    if the piece on the from-square is actually a pawn. Any other piece
    (rook, bishop, queen) sliding to the back rank uses the exact same
    plane, since the encoding is purely geometric. Without checking the
    piece type here, decode_index would tag a plain rook/queen slide to
    the back rank with promotion=QUEEN -- and python-chess's Board.push()
    unconditionally honors move.promotion regardless of what piece is
    actually moving, silently turning e.g. a rook into a queen on the
    board. `board` is required precisely to make this check.
    """
    white_to_move = board.turn == chess.WHITE
    from_square, plane = divmod(index, PLANES_PER_SQUARE)
    from_rank, from_file = divmod(from_square, 8)

    # Real (unoriented) from-square, so we can look up the actual piece.
    # Need to unmirror before checking the piece type, since the board is in real coordinates.
    real_from_square = from_square if white_to_move else chess.square_mirror(from_square)
    moving_piece_type = board.piece_type_at(real_from_square)

    #default value for promotion is None, only set to chess.QUEEN or an underpromotion piece if the move is actually a promotion
    promotion = None
    if plane < 56:
        direction_index, dist_minus_one = divmod(plane, 7)
        df, dr = QUEEN_DIRECTIONS[direction_index]
        distance = dist_minus_one + 1
        to_file = from_file + df * distance
        to_rank = from_rank + dr * distance
        # Only treat this as a queen promotion if a pawn is actually the
        # piece making the move -- otherwise this is a normal queen/rook/
        # bishop slide that happens to land on the back rank.
        if to_rank == 7 and moving_piece_type == chess.PAWN:
            promotion = chess.QUEEN
    elif plane < 64:
        df, dr = KNIGHT_DELTAS[plane - 56]
        to_file = from_file + df
        to_rank = from_rank + dr
    else:
        sub = plane - 64
        direction_index, piece_index = divmod(sub, 3)
        df = UNDERPROMO_DIRECTIONS[direction_index]
        to_file = from_file + df
        to_rank = from_rank + 1
        promotion = UNDERPROMO_PIECES[piece_index]

    if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
        return None

    to_square = to_rank * 8 + to_file
    oriented_move = chess.Move(from_square, to_square, promotion=promotion)
    return _orient_move(oriented_move, white_to_move)  # mirror is its own inverse


def legal_move_mask(board: chess.Board) -> list[int]:
    """
    Return the list of policy indices corresponding to every legal move on
    `board`. Useful for masking illegal moves out of the policy head's
    logits before applying softmax, and for building policy training
    targets from a (board, played_move) pair.
    """
    return [encode_move(board, move) for move in board.legal_moves]


# sanity checks
if __name__ == "__main__":
    # round-trip check: every legal move from the start position
    board = chess.Board()
    for move in board.legal_moves:
        idx = encode_move(board, move)
        decoded = decode_index(board, idx)
        # tests if the condition is true using assert statement, prints failed message if the condition is false
        assert decoded == move, f"round-trip failed: {move} -> {idx} -> {decoded}"
    print(f"start position: {len(list(board.legal_moves))} legal moves, all round-trip correctly")

    # round-trip check with Black to move (exercises the mirroring path)
    board.push_san("e4")
    for move in board.legal_moves:
        idx = encode_move(board, move)
        decoded = decode_index(board, idx)
        assert decoded == move, f"round-trip failed (Black to move): {move} -> {idx} -> {decoded}"
    print(f"after 1.e4, Black to move: {len(list(board.legal_moves))} legal moves, all round-trip correctly")

    # round-trip check on a position with underpromotion options available
    #FEN notation: https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation
    promo_board = chess.Board("8/P7/8/8/8/8/8/k1K5 w - - 0 1")
    for move in promo_board.legal_moves:
        idx = encode_move(promo_board, move)
        decoded = decode_index(promo_board, idx)
        assert decoded == move, f"round-trip failed (promotion): {move} -> {idx} -> {decoded}"
    promo_moves = [m for m in promo_board.legal_moves if m.promotion is not None]
    print(f"promotion position: {len(promo_moves)} promotion moves (queen + underpromotions), all round-trip correctly")

    # regression check: a non-pawn piece sliding to the back rank must not be mistaken for a queen promotion
    rook_board = chess.Board("8/8/8/8/8/8/6k1/R3K3 w - - 0 1")
    rook_move = chess.Move.from_uci("a1a8")
    assert rook_move in rook_board.legal_moves
    idx = encode_move(rook_board, rook_move)
    decoded = decode_index(rook_board, idx)
    assert decoded == rook_move, f"back-rank rook slide misdecoded: {decoded}"
    assert decoded.promotion is None, "rook slide to the back rank must not carry a promotion"
    after_push = rook_board.copy()
    after_push.push(decoded)
    assert after_push.piece_type_at(chess.A8) == chess.ROOK, "piece was corrupted into a queen on push"
    print("back-rank non-pawn slide: correctly decoded with no promotion, piece identity preserved")

    # same regression check from Black's side (exercises the mirrored from-square lookup)
    rook_board_b = chess.Board("r3k3/6K1/8/8/8/8/8/8 b - - 0 1")
    rook_move_b = chess.Move.from_uci("a8a1")
    assert rook_move_b in rook_board_b.legal_moves
    idx_b = encode_move(rook_board_b, rook_move_b)
    decoded_b = decode_index(rook_board_b, idx_b)
    assert decoded_b == rook_move_b, f"back-rank rook slide (Black) misdecoded: {decoded_b}"
    assert decoded_b.promotion is None
    print("back-rank non-pawn slide (Black to move): correctly decoded with no promotion")

    # legal_move_mask sanity check
    mask = legal_move_mask(chess.Board())
    assert len(mask) == 20  # 20 legal moves in the start position
    assert len(set(mask)) == 20  # all indices distinct
    print("legal_move_mask: 20 unique indices for the start position, as expected")

    print("all move-encoding sanity checks passed")