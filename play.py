import sys
import chess
import pygame
import torch
from network_model import ChessNet
from dataset import board_to_tensor
from engine_output_dataset import encode_move

MODEL_PATH = r"C:\Projects\Chess-Engine-Project\model.pt" #use your own path to the model checkpoint here

SQUARE = 72
BOARD_PX = SQUARE * 8
PANEL_W = 420 
WIN_W = BOARD_PX + PANEL_W
WIN_H = BOARD_PX

LIGHT = (238, 238, 210)
DARK = (118, 150, 86)
SELECT = (246, 246, 105)
DEST_DOT = (60, 60, 60)
PANEL_BG = (30, 30, 30)
TEXT_COL = (230, 230, 230)
WARN_COL = (255, 210, 90)
BTN_BG = (60, 60, 60)
BTN_HOVER = (85, 85, 85)

# Solid glyphs used for stroke rendering
UNICODE_PIECES = {
    "P": "\u265F", "N": "\u265E", "B": "\u265D", "R": "\u265C", "Q": "\u265B", "K": "\u265A",
    "p": "\u265F", "n": "\u265E", "b": "\u265D", "r": "\u265C", "q": "\u265B", "k": "\u265A",
}


def load_engine(path):
    print(f"loading checkpoint: {path}")
    net = ChessNet.load_checkpoint(path)
    print(f"loaded on device: {next(net.parameters()).device}")
    return net


def legal_move_map(board):
    mapping = {}
    for move in board.legal_moves:
        mapping[encode_move(board, move)] = move
    return mapping


def engine_choose_move(net, board, temperature=0.0):
    move_map = legal_move_map(board)
    indices = list(move_map.keys())
    tensor = board_to_tensor(board)
    probs, value = net.infer(tensor, indices)
    if temperature <= 0:
        best_idx = max(indices, key=lambda i: probs[i].item())
    else:
        weights = torch.tensor([probs[i].item() ** (1.0 / temperature) for i in indices])
        weights = weights / weights.sum()
        best_idx = indices[torch.multinomial(weights, 1).item()]
    return move_map[best_idx], value


class Button:
    def __init__(self, rect, label):
        self.rect = pygame.Rect(rect)
        self.label = label

    def draw(self, screen, font, hovered):
        pygame.draw.rect(screen, BTN_HOVER if hovered else BTN_BG, self.rect, border_radius=6)
        text = font.render(self.label, True, TEXT_COL)
        screen.blit(text, text.get_rect(center=self.rect.center))

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


def square_to_xy(square, flipped):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    if not flipped:
        col, row = file, 7 - rank
    else:
        col, row = 7 - file, rank
    return col * SQUARE, row * SQUARE


def xy_to_square(x, y, flipped):
    col, row = x // SQUARE, y // SQUARE
    if not flipped:
        file, rank = col, 7 - row
    else:
        file, rank = 7 - col, row
    return chess.square(file, rank)


def render_stroked_text(font, text, main_color, outline_color, outline_width=2):
    """Renders text with a thick outline around it."""
    base = font.render(text, True, main_color)
    outline = font.render(text, True, outline_color)
    
    w = base.get_width() + 2 * outline_width
    h = base.get_height() + 2 * outline_width
    surf = pygame.Surface((w, h), pygame.SRCALPHA)

    # Draw outline text at surrounding offsets
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                surf.blit(outline, (dx + outline_width, dy + outline_width))

    # Draw central fill text
    surf.blit(base, (outline_width, outline_width))
    return surf


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else MODEL_PATH
    net = load_engine(model_path)

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Thinking Monkey Engine")
    clock = pygame.time.Clock()

    piece_font = pygame.font.SysFont("segoe ui symbol", SQUARE - 14) or pygame.font.SysFont(None, SQUARE - 14)
    ui_font = pygame.font.SysFont("consolas", 18)
    small_font = pygame.font.SysFont("consolas", 15)
    big_font = pygame.font.SysFont("consolas", 22, bold=True)
    coord_font = pygame.font.SysFont("consolas", 14, bold=True)

    state = {
        "board": chess.Board(),
        "san_history": [],
        "selected_square": None,
        "legal_dest": {},
        "flipped": False,
        "human_color": chess.WHITE,
        "last_value": None,
        "status_msg": "",
        "pending_promo": None,
    }

    new_game_btn = Button((BOARD_PX + 15, 20, PANEL_W - 30, 36), "New Game")
    swap_btn = Button((BOARD_PX + 15, 64, PANEL_W - 30, 36), "Play Other Side")

    def reset_game():
        state["board"] = chess.Board()
        state["san_history"] = []
        state["selected_square"] = None
        state["legal_dest"] = {}
        state["last_value"] = None
        state["status_msg"] = ""
        state["pending_promo"] = None

    def do_engine_move():
        board = state["board"]
        if board.is_game_over():
            return
        state["status_msg"] = "engine thinking..."
        redraw()
        pygame.display.flip()
        move, value = engine_choose_move(net, board, temperature=0.0)
        state["san_history"].append(board.san(move))
        board.push(move)
        state["last_value"] = value
        state["status_msg"] = ""

    def game_status_text():
        board = state["board"]
        if board.is_checkmate():
            winner = "Black" if board.turn == chess.WHITE else "White"
            return f"Checkmate -- {winner} wins"
        if board.is_stalemate():
            return "Draw (stalemate)"
        if board.is_insufficient_material():
            return "Draw (insufficient material)"
        if board.can_claim_fifty_moves():
            return "Draw (50-move rule)"
        if board.can_claim_threefold_repetition():
            return "Draw (threefold repetition)"
        if board.is_check():
            return "Check!"
        return ""

    def draw_promo_menu():
        from_sq, to_sq, moves_by_piece = state["pending_promo"]
        x, y = square_to_xy(to_sq, state["flipped"])
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for i, pt in enumerate(pieces):
            top = y + i * SQUARE if y + i * SQUARE + SQUARE <= WIN_H else y - (i + 1) * SQUARE
            box = pygame.Rect(x, top, SQUARE, SQUARE)
            pygame.draw.rect(screen, (255, 255, 255), box)
            pygame.draw.rect(screen, (0, 0, 0), box, 2)
            symbol = chess.Piece(pt, state["human_color"]).symbol()
            fill_col = (255, 255, 255) if state["human_color"] == chess.WHITE else (75, 75, 75)
            text = render_stroked_text(piece_font, UNICODE_PIECES[symbol], fill_col, (0, 0, 0), 2)
            screen.blit(text, text.get_rect(center=box.center))

    def promo_menu_click(pos):
        from_sq, to_sq, moves_by_piece = state["pending_promo"]
        x, y = square_to_xy(to_sq, state["flipped"])
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for i, pt in enumerate(pieces):
            top = y + i * SQUARE if y + i * SQUARE + SQUARE <= WIN_H else y - (i + 1) * SQUARE
            box = pygame.Rect(x, top, SQUARE, SQUARE)
            if box.collidepoint(pos):
                return moves_by_piece[pt]
        return None

    def draw_board_labels():
        flipped = state["flipped"]
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']

        if flipped:
            files = files[::-1]
            ranks = ranks[::-1]

        for col in range(8):
            # File labels drawn inside the bottom row of squares
            sq_color = DARK if col % 2 == 0 else LIGHT
            txt_color = LIGHT if sq_color == DARK else DARK
            file_lbl = coord_font.render(files[col], True, txt_color)
            lbl_x = col * SQUARE + SQUARE - file_lbl.get_width() - 4
            lbl_y = BOARD_PX - file_lbl.get_height() - 2
            screen.blit(file_lbl, (lbl_x, lbl_y))

        for row in range(8):
            # Rank labels drawn inside the leftmost column of squares
            sq_color = LIGHT if row % 2 == 0 else DARK
            txt_color = DARK if sq_color == LIGHT else LIGHT
            rank_lbl = coord_font.render(ranks[7 - row], True, txt_color)
            screen.blit(rank_lbl, (4, row * SQUARE + 3))

    def redraw():
        board = state["board"]
        flipped = state["flipped"]
        screen.fill(PANEL_BG)

        for square in chess.SQUARES:
            x, y = square_to_xy(square, flipped)
            color = LIGHT if (chess.square_file(square) + chess.square_rank(square)) % 2 != 0 else DARK
            if square == state["selected_square"]:
                color = SELECT
            pygame.draw.rect(screen, color, (x, y, SQUARE, SQUARE))
            
            piece = board.piece_at(square)
            if piece:
                glyph = UNICODE_PIECES[piece.symbol()]
                if piece.color == chess.WHITE:
                    text_surf = render_stroked_text(piece_font, glyph, (255, 255, 255), (0, 0, 0), outline_width=2)
                else:
                    text_surf = render_stroked_text(piece_font, glyph, (75, 75, 75), (0, 0, 0), outline_width=2)
                
                screen.blit(text_surf, text_surf.get_rect(center=(x + SQUARE // 2, y + SQUARE // 2)))

        draw_board_labels()

        for sq in state["legal_dest"]:
            x, y = square_to_xy(sq, flipped)
            pygame.draw.circle(screen, DEST_DOT, (x + SQUARE // 2, y + SQUARE // 2), 10)

        mouse = pygame.mouse.get_pos()
        new_game_btn.draw(screen, ui_font, new_game_btn.clicked(mouse))
        swap_btn.draw(screen, ui_font, swap_btn.clicked(mouse))

        y0 = 125
        you_are = "White" if state["human_color"] == chess.WHITE else "Black"
        screen.blit(big_font.render(f"You are: {you_are}", True, TEXT_COL), (BOARD_PX + 15, y0))
        if state["last_value"] is not None:
            perspective = "White" if board.turn == chess.WHITE else "Black"
            val_text = f"Evaluation from the monkey's perspective: {state['last_value']:+.3f}"
            screen.blit(small_font.render(val_text, True, TEXT_COL), (BOARD_PX + 15, y0 + 34))

        status = state["status_msg"] or game_status_text()
        if status:
            screen.blit(small_font.render(status, True, WARN_COL), (BOARD_PX + 15, y0 + 60))

        if state["pending_promo"]:
            draw_promo_menu()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos

                if new_game_btn.clicked(pos):
                    reset_game()
                    continue
                if swap_btn.clicked(pos):
                    state["human_color"] = not state["human_color"]
                    state["flipped"] = state["human_color"] == chess.BLACK
                    reset_game()
                    continue

                if state["pending_promo"]:
                    move = promo_menu_click(pos)
                    if move:
                        board = state["board"]
                        state["san_history"].append(board.san(move))
                        board.push(move)
                        state["pending_promo"] = None
                        state["selected_square"] = None
                        state["legal_dest"] = {}
                    continue

                if pos[0] >= BOARD_PX or pos[1] >= WIN_H:
                    continue

                board = state["board"]
                if board.is_game_over() or board.turn != state["human_color"]:
                    continue

                clicked_sq = xy_to_square(pos[0], pos[1], state["flipped"])
                selected_square = state["selected_square"]

                if selected_square is None:
                    piece = board.piece_at(clicked_sq)
                    if piece and piece.color == state["human_color"]:
                        state["selected_square"] = clicked_sq
                        state["legal_dest"] = {m.to_square: m for m in board.legal_moves if m.from_square == clicked_sq}
                else:
                    if clicked_sq == selected_square:
                        state["selected_square"] = None
                        state["legal_dest"] = {}
                    elif clicked_sq in state["legal_dest"]:
                        promo_moves = [
                            m for m in board.legal_moves
                            if m.from_square == selected_square and m.to_square == clicked_sq and m.promotion
                        ]
                        if promo_moves:
                            state["pending_promo"] = (selected_square, clicked_sq, {m.promotion: m for m in promo_moves})
                        else:
                            move = state["legal_dest"][clicked_sq]
                            state["san_history"].append(board.san(move))
                            board.push(move)
                        state["selected_square"] = None
                        state["legal_dest"] = {}
                    else:
                        piece = board.piece_at(clicked_sq)
                        if piece and piece.color == state["human_color"]:
                            state["selected_square"] = clicked_sq
                            state["legal_dest"] = {m.to_square: m for m in board.legal_moves if m.from_square == clicked_sq}
                        else:
                            state["selected_square"] = None
                            state["legal_dest"] = {}

        redraw()
        pygame.display.flip()

        board = state["board"]
        if not board.is_game_over() and board.turn != state["human_color"] and not state["pending_promo"]:
            do_engine_move()

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()