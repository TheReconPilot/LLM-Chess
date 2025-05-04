import os
import sys
import re
import time
import math
import random
import threading
import subprocess
import pygame
import chess
import chess.pgn
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import pyperclip  # for copy-to-clipboard functionality
from dotenv import load_dotenv

load_dotenv("../.env")
pygame.init()

STOCKFISH_PATH = os.environ['stockfish_path']
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ---------------------------
# SCREEN SIZING AND SCALING
# ---------------------------
temp_screen = pygame.display.set_mode((1, 1))
info = pygame.display.Info()
screen_width, screen_height = info.current_w, info.current_h

# Use 95% width and 90% height of available screen space
desired_width = int(screen_width * 0.95)
desired_height = int(screen_height * 0.90)

# Original design dimensions
ORIGINAL_WIDTH = 760
ORIGINAL_HEIGHT = 820

scale = min(desired_width / ORIGINAL_WIDTH, desired_height / ORIGINAL_HEIGHT)
if scale <= 0:
    scale = 1

# Scaled layout constants
EVAL_BAR_WIDTH       = int(40 * scale)
BOARD_SIZE           = int(560 * scale)
TOP_PANEL_HEIGHT     = int(30 * scale)
BOTTOM_PANEL_HEIGHT  = int(30 * scale)
SIDE_PANEL_WIDTH     = int(160 * scale)
HISTORY_PANEL_HEIGHT = int(200 * scale)

WIDTH  = EVAL_BAR_WIDTH + BOARD_SIZE + SIDE_PANEL_WIDTH
HEIGHT = TOP_PANEL_HEIGHT + BOARD_SIZE + BOTTOM_PANEL_HEIGHT + HISTORY_PANEL_HEIGHT

SQUARE_SIZE = BOARD_SIZE // 8

print("Calculated window dimensions:", WIDTH, "x", HEIGHT)
print("SQUARE_SIZE:", SQUARE_SIZE)

# ---------------------------
# COLORS & STOCKFISH PATH
# ---------------------------
WHITE      = (240, 217, 181)
BLACK      = (181, 136, 99)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY  = (50, 50, 50)
GREEN      = (0, 255, 0)
RED        = (255, 0, 0)
BLUE       = (0, 0, 255)  # For selection highlight

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------
difficulty = 1600
show_eval = True
show_best_arrow = True
show_threat_arrow = True

best_move_arrow = None
threat_move_arrow = None
arrow_update_interval = 3
last_arrow_update = 0
updating_arrows = False

current_fen = None
eval_score = 0
eval_text = "0"

# Modes: "otb" (Over the Board), "computer" (Stockfish), or "llm" (LLM-based)
play_mode = "otb"
computer_color = None

history_scroll = 0
eval_history = []  # Record eval (in cp) after each move
game_over_shown = False

# ---------------------------
# SIDE PANEL BUTTON RECTS
# ---------------------------
SIDE_PANEL_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE,
    TOP_PANEL_HEIGHT,
    SIDE_PANEL_WIDTH,
    BOARD_SIZE
)
BUTTON_MARGIN = int(10 * scale)
BUTTON_WIDTH = SIDE_PANEL_WIDTH - 2 * BUTTON_MARGIN
BUTTON_HEIGHT = int(40 * scale)

FLIP_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
PREV_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 60,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
NEXT_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 120,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
TAKEBACK_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 180,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
EVAL_TOGGLE_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 240,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
BEST_ARROW_TOGGLE_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 300,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
THREAT_ARROW_TOGGLE_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 360,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
DIFF_UP_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 420,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)
DIFF_DOWN_BUTTON_RECT = pygame.Rect(
    EVAL_BAR_WIDTH + BOARD_SIZE + BUTTON_MARGIN,
    TOP_PANEL_HEIGHT + BUTTON_MARGIN + 480,
    BUTTON_WIDTH,
    BUTTON_HEIGHT
)

# ---------------------------
# LOAD CHESS PIECE IMAGES
# ---------------------------
def load_scaled_pieces():
    pieces = {}
    for p in ['p', 'r', 'n', 'b', 'q', 'k']:
        pieces[f"b{p}"] = pygame.transform.scale(
            pygame.image.load(f"images/b{p}.png"), (SQUARE_SIZE, SQUARE_SIZE)
        )
        pieces[f"w{p}"] = pygame.transform.scale(
            pygame.image.load(f"images/w{p}.png"), (SQUARE_SIZE, SQUARE_SIZE)
        )
    return pieces

PIECES = load_scaled_pieces()

# ---------------------------
# INITIALIZE LLM PIPELINE
# ---------------------------
generator = pipeline("text-generation", model=MODEL_NAME)

# ---------------------------
# STOCKFISH MOVE SUGGESTION
# ---------------------------
def get_suggestion_for_color(board, color, apply_difficulty=False):
    global difficulty
    try:
        proc = subprocess.Popen(
            [STOCKFISH_PATH],
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        fen_parts = board.fen().split(" ")
        fen_parts[1] = "w" if color == chess.WHITE else "b"
        mod_fen = " ".join(fen_parts)

        proc.stdin.write("uci\n")
        proc.stdin.flush()
        while proc.stdout.readline().strip() != "uciok":
            pass

        skill = 20
        if apply_difficulty:
            skill = round((difficulty - 100) / 3100 * 20)
            skill = max(0, min(skill, 20))
        proc.stdin.write(f"setoption name Skill Level value {skill}\n")
        proc.stdin.write("isready\n")
        proc.stdin.flush()
        while proc.stdout.readline().strip() != "readyok":
            pass

        proc.stdin.write(f"position fen {mod_fen}\n")
        proc.stdin.write("go movetime 100\n")
        proc.stdin.flush()

        bestmove = None
        while True:
            line = proc.stdout.readline().strip()
            if line.startswith("bestmove"):
                bestmove = line.split()[1]
                break
        proc.stdin.write("quit\n")
        proc.stdin.flush()
        proc.terminate()
        if bestmove == "(none)":
            return None
        return chess.Move.from_uci(bestmove)
    except Exception as e:
        print("Error getting suggestion:", e)
        return None

# ---------------------------
# LLM MOVE GENERATION
# ---------------------------

def _extract_moves_from_response(response):
        move_pattern = r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O|O-O-O)\b"
        matches = re.findall(move_pattern, response)
        return [m.strip() for m in matches]


def get_llm_move(board, color):
    fen = board.fen()
    color_str = "White" if color == chess.WHITE else "Black"
    prompt = (
        f"You are a chess AI. Given the following board position in FEN format:\n"
        f"{fen}\n"
        f"Please provide your next move in UCI notation for {color_str} (e.g. e2e4). "
        f"Output only the move in UCI notation."
    )
    try:
        result = generator(prompt, max_length=128, do_sample=True, truncation=True, temperature=0.7)
        text = result[0]["generated_text"]
        move_str = _extract_moves_from_response(text)[-1]
        print("LLM suggests:", move_str)
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
        else:
            print("Illegal move from LLM; choosing random move.")
            return random.choice(list(board.legal_moves))
    except Exception as e:
        print("Error from LLM:", e)
        return random.choice(list(board.legal_moves))

# ---------------------------
# UPDATE ARROWS THREAD
# ---------------------------
def update_arrows_thread(board_copy, ai_mode, ai_color, flipped):
    global best_move_arrow, threat_move_arrow, last_arrow_update, updating_arrows
    if ai_mode in ["computer", "llm"]:
        human_color_local = not ai_color
        comp_color_local = ai_color
    else:
        human_color_local = chess.WHITE if not flipped else chess.BLACK
        comp_color_local = chess.BLACK if not flipped else chess.WHITE

    best_move_arrow = get_suggestion_for_color(board_copy, human_color_local, apply_difficulty=False)
    threat_move_arrow = get_suggestion_for_color(board_copy, comp_color_local, apply_difficulty=True)
    last_arrow_update = time.time()
    updating_arrows = False

# ---------------------------
# STOCKFISH EVALUATION LOOP
# ---------------------------
def stockfish_evaluation_loop(proc):
    """
    Force the evaluation from White’s perspective so that the sign remains consistent.
    """
    global current_fen, eval_score, eval_text
    proc.stdin.write("uci\n")
    proc.stdin.flush()
    while proc.stdout.readline().strip() != "uciok":
        pass

    max_eval = 1000
    while True:
        if current_fen:
            # Force the FEN to have White to move
            fen_parts = current_fen.split()
            if len(fen_parts) > 1:
                fen_parts[1] = "w"
            forced_fen = " ".join(fen_parts)
            proc.stdin.write(f"position fen {forced_fen}\n")
            proc.stdin.write("go movetime 100\n")
            proc.stdin.flush()

            new_eval_cp = None
            new_eval_text = None
            while True:
                line = proc.stdout.readline().strip()
                if line.startswith("info"):
                    parts = line.split()
                    if "score" in parts:
                        idx = parts.index("score")
                        if idx + 2 < len(parts):
                            if parts[idx+1] == "cp":
                                try:
                                    cp_val = int(parts[idx+2])
                                    new_eval_cp = cp_val
                                    new_eval_text = str(cp_val)
                                except:
                                    pass
                            elif parts[idx+1] == "mate":
                                try:
                                    mate_val = int(parts[idx+2])
                                    new_eval_text = "M" + str(abs(mate_val))
                                    new_eval_cp = max_eval if mate_val > 0 else -max_eval
                                except:
                                    pass
                if line.startswith("bestmove"):
                    break
            if new_eval_cp is not None and new_eval_text is not None:
                eval_score = new_eval_cp
                eval_text = new_eval_text
        time.sleep(0.1)

# ---------------------------
# DRAWING FUNCTIONS
# ---------------------------
def draw_evaluation_bar(screen):
    bar_rect = pygame.Rect(0, TOP_PANEL_HEIGHT, EVAL_BAR_WIDTH, BOARD_SIZE)
    pygame.draw.rect(screen, LIGHT_GRAY, bar_rect)
    max_eval = 1000
    proportion = max(min(eval_score, max_eval), -max_eval) / max_eval
    bar_center = TOP_PANEL_HEIGHT + BOARD_SIZE // 2
    max_fill = BOARD_SIZE // 2
    fill_pixels = int(abs(proportion) * max_fill)
    if proportion > 0:
        fill_rect = pygame.Rect(0, bar_center - fill_pixels, EVAL_BAR_WIDTH, fill_pixels)
        pygame.draw.rect(screen, (255,255,255), fill_rect)
    elif proportion < 0:
        fill_rect = pygame.Rect(0, bar_center, EVAL_BAR_WIDTH, fill_pixels)
        pygame.draw.rect(screen, (0,0,0), fill_rect)
    pygame.draw.line(screen, (100,100,100), (0,bar_center), (EVAL_BAR_WIDTH,bar_center), 2)
    pygame.draw.rect(screen, (0,0,0), bar_rect, 2)
    font = pygame.font.Font(None, 20)
    text_color = (0,0,0) if proportion > 0 else (255,255,255)
    text_surface = font.render(eval_text, True, text_color)
    text_rect = text_surface.get_rect(center=(EVAL_BAR_WIDTH//2, bar_center))
    screen.blit(text_surface, text_rect)

def draw_board(screen, flipped):
    for row in range(8):
        for col in range(8):
            draw_row = 7 - row if flipped else row
            draw_col = 7 - col if flipped else col
            color = WHITE if (draw_row + draw_col) % 2 == 0 else BLACK
            rect = pygame.Rect(EVAL_BAR_WIDTH + col * SQUARE_SIZE,
                               TOP_PANEL_HEIGHT + row * SQUARE_SIZE,
                               SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

def draw_pieces(screen, board, flipped):
    for row in range(8):
        for col in range(8):
            sq = chess.square(7 - col, row) if flipped else chess.square(col, 7 - row)
            piece = board.piece_at(sq)
            if piece:
                c = 'w' if piece.color == chess.WHITE else 'b'
                img = PIECES[f"{c}{piece.symbol().lower()}"]
                screen.blit(img, (EVAL_BAR_WIDTH + col * SQUARE_SIZE,
                                  TOP_PANEL_HEIGHT + row * SQUARE_SIZE))

def draw_side_panel(screen):
    font = pygame.font.Font(None, 24)
    pygame.draw.rect(screen, (220,220,220), SIDE_PANEL_RECT)
    def draw_button(rect, text, text_color=(255,255,255)):
        pygame.draw.rect(screen, (100,100,100), rect)
        surf = font.render(text, True, text_color)
        screen.blit(surf, surf.get_rect(center=rect.center))
    draw_button(FLIP_BUTTON_RECT, "Flip Board")
    draw_button(PREV_BUTTON_RECT, "Previous")
    draw_button(NEXT_BUTTON_RECT, "Next")
    draw_button(TAKEBACK_BUTTON_RECT, "Takeback")
    eval_toggle_text = "Hide Eval" if show_eval else "Show Eval"
    draw_button(EVAL_TOGGLE_BUTTON_RECT, eval_toggle_text)
    best_toggle_text = "Hide Best" if show_best_arrow else "Show Best"
    draw_button(BEST_ARROW_TOGGLE_BUTTON_RECT, best_toggle_text)
    threat_toggle_text = "Hide Threat" if show_threat_arrow else "Show Threat"
    draw_button(THREAT_ARROW_TOGGLE_BUTTON_RECT, threat_toggle_text)
    draw_button(DIFF_UP_BUTTON_RECT, "Diff +")
    draw_button(DIFF_DOWN_BUTTON_RECT, "Diff -")
    diff_disp = font.render(f"Diff: {difficulty}", True, (0,0,0))
    diff_rect = diff_disp.get_rect(center=(DIFF_UP_BUTTON_RECT.centerx, DIFF_UP_BUTTON_RECT.bottom+15))
    screen.blit(diff_disp, diff_rect)

def draw_move_history_panel(screen, board, scroll):
    panel_rect = pygame.Rect(EVAL_BAR_WIDTH, TOP_PANEL_HEIGHT + BOARD_SIZE + BOTTOM_PANEL_HEIGHT,
                               BOARD_SIZE, HISTORY_PANEL_HEIGHT)
    pygame.draw.rect(screen, (230,230,230), panel_rect)
    pygame.draw.rect(screen, (0,0,0), panel_rect, 2)
    font = pygame.font.Font(None, 20)
    temp_board = chess.Board()
    moves = []
    for mv in board.move_stack:
        moves.append(temp_board.san(mv))
        temp_board.push(mv)
    lines = [f"{(i//2)+1}. {moves[i]}" + (f" {moves[i+1]}" if i+1 < len(moves) else "") 
             for i in range(0, len(moves), 2)]
    line_height = font.get_height() + 4
    total_text_height = line_height * len(lines)
    text_surf = pygame.Surface((panel_rect.width - 10, total_text_height))
    text_surf.fill((230,230,230))
    for idx, ln in enumerate(lines):
        ln_surf = font.render(ln, True, (0,0,0))
        text_surf.blit(ln_surf, (0, idx*line_height))
    screen.blit(text_surf, (panel_rect.x+5, panel_rect.y+5-scroll))
    if total_text_height > panel_rect.height - 10:
        scrollbar_height = (panel_rect.height - 10) * (panel_rect.height - 10) / total_text_height
        scrollbar_y = panel_rect.y + 5 + (panel_rect.height - 10 - scrollbar_height) * (scroll/(total_text_height - (panel_rect.height - 10)))
        pygame.draw.rect(screen, DARK_GRAY, (panel_rect.right-10, panel_rect.y+5, 8, scrollbar_height))

def draw_arrow(screen, start, end, color, thickness=5):
    pygame.draw.line(screen, color, start, end, thickness)
    dx, dy = end[0]-start[0], end[1]-start[1]
    angle = math.atan2(dy, dx)
    arrow_length = 15
    arrow_angle = math.pi/6
    x1 = end[0] - arrow_length * math.cos(angle - arrow_angle)
    y1 = end[1] - arrow_length * math.sin(angle - arrow_angle)
    x2 = end[0] - arrow_length * math.cos(angle + arrow_angle)
    y2 = end[1] - arrow_length * math.sin(angle + arrow_angle)
    pygame.draw.polygon(screen, color, [end, (x1,y1), (x2,y2)])

def square_to_coords(sq, flipped):
    f = chess.square_file(sq)
    r = chess.square_rank(sq)
    if not flipped:
        col, row = f, 7 - r
    else:
        col, row = 7 - f, r
    return EVAL_BAR_WIDTH + col * SQUARE_SIZE, TOP_PANEL_HEIGHT + row * SQUARE_SIZE

def draw_move_arrow(screen, mv, flipped, color):
    start = square_to_coords(mv.from_square, flipped)
    end = square_to_coords(mv.to_square, flipped)
    start = (start[0] + SQUARE_SIZE//2, start[1] + SQUARE_SIZE//2)
    end = (end[0] + SQUARE_SIZE//2, end[1] + SQUARE_SIZE//2)
    draw_arrow(screen, start, end, color)

def draw_last_move_highlight(screen, board, flipped):
    if board.move_stack:
        last_mv = board.move_stack[-1]
        highlight_color = (0,255,0)
        for sq in [last_mv.from_square, last_mv.to_square]:
            x, y = square_to_coords(sq, flipped)
            rect = pygame.Rect(x, y, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, highlight_color, rect, 4)

def draw_selected_highlight(screen, board, selected_square, flipped):
    if selected_square is None:
        return
    x, y = square_to_coords(selected_square, flipped)
    pygame.draw.rect(screen, BLUE, (x, y, SQUARE_SIZE, SQUARE_SIZE), 4)
    for mv in board.legal_moves:
        if mv.from_square == selected_square:
            dx, dy = square_to_coords(mv.to_square, flipped)
            if board.piece_at(mv.to_square) is None:
                center = (dx + SQUARE_SIZE//2, dy + SQUARE_SIZE//2)
                pygame.draw.circle(screen, BLUE, center, SQUARE_SIZE//6)
            else:
                pygame.draw.rect(screen, BLUE, (dx, dy, SQUARE_SIZE, SQUARE_SIZE), 4)

def get_captured_pieces(board):
    """
    Return two lists: (white_caps, black_caps) of pieces captured by White/Black.
    White's captured list means black pieces that White has captured, etc.
    """
    initial = {'p':8, 'n':2, 'b':2, 'r':2, 'q':1}
    white_caps = []
    black_caps = []
    for piece in ['p','n','b','r','q']:
        count_black = sum(1 for p in board.piece_map().values() if p.color==chess.BLACK and p.symbol()==piece)
        captured_black = initial[piece] - count_black
        white_caps.extend([piece]*max(captured_black,0))
    for piece in ['p','n','b','r','q']:
        count_white = sum(1 for p in board.piece_map().values() if p.color==chess.WHITE and p.symbol()==piece.upper())
        captured_white = initial[piece] - count_white
        black_caps.extend([piece.upper()]*max(captured_white,0))
    return white_caps, black_caps

def draw_captured_panels(screen, board):
    """
    Draw top panel for White's captured pieces (i.e. black pieces White took)
    and bottom panel for Black's captured pieces (white pieces Black took).
    """
    white_caps, black_caps = get_captured_pieces(board)
    panel_height = TOP_PANEL_HEIGHT
    margin = 5
    icon_size = 32

    # Top panel
    top_rect = pygame.Rect(EVAL_BAR_WIDTH, 0, BOARD_SIZE, panel_height)
    pygame.draw.rect(screen, LIGHT_GRAY, top_rect)
    x = EVAL_BAR_WIDTH + margin
    y = (panel_height - icon_size)//2
    for pc in white_caps:
        key = f"b{pc}"
        if key in PIECES:
            sc = pygame.transform.scale(PIECES[key], (icon_size, icon_size))
            screen.blit(sc, (x, y))
        x += icon_size + margin

    # Bottom panel
    bottom_rect = pygame.Rect(EVAL_BAR_WIDTH, TOP_PANEL_HEIGHT+BOARD_SIZE, BOARD_SIZE, panel_height)
    pygame.draw.rect(screen, LIGHT_GRAY, bottom_rect)
    x = EVAL_BAR_WIDTH + margin
    y = bottom_rect.y + (panel_height - icon_size)//2
    for pc in black_caps:
        key = f"w{pc.lower()}"
        if key in PIECES:
            sc = pygame.transform.scale(PIECES[key], (icon_size, icon_size))
            screen.blit(sc, (x, y))
        x += icon_size + margin

    # Show material difference
    values = {'p':1, 'n':3, 'b':3, 'r':5, 'q':9}
    w_mat = sum(values[p] for p in white_caps)
    b_mat = sum(values[p.lower()] for p in black_caps)
    lead = w_mat - b_mat
    font = pygame.font.Font(None, 24)
    if lead != 0:
        lead_text = font.render(f"+{abs(lead)}", True, (0,0,0))
        rct = lead_text.get_rect()
        if lead > 0:
            rct.right = EVAL_BAR_WIDTH + BOARD_SIZE - margin
            rct.centery = top_rect.centery
        else:
            rct.right = EVAL_BAR_WIDTH + BOARD_SIZE - margin
            rct.centery = bottom_rect.centery
        screen.blit(lead_text, rct)

# ---------------------------
# SHOW PGN DIALOG WITH COPY
# ---------------------------
def show_pgn_dialog(screen, board):
    # Generate PGN text using a StringExporter
    game = chess.pgn.Game.from_board(board)
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_text = game.accept(exporter)

    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    box_width = 600
    box_height = 400
    box_x = (WIDTH - box_width) // 2
    box_y = (HEIGHT - box_height) // 2
    dialog_rect = pygame.Rect(box_x, box_y, box_width, box_height)
    pygame.draw.rect(screen, (240,240,240), dialog_rect)
    pygame.draw.rect(screen, (0,0,0), dialog_rect, 2)

    font_large = pygame.font.Font(None, 32)
    title = font_large.render("Game PGN (copy below):", True, (0,0,0))
    screen.blit(title, title.get_rect(center=(box_x+box_width//2, box_y+30)))

    # Render PGN text. For simplicity, split by newline.
    font_small = pygame.font.Font(None, 24)
    lines = pgn_text.splitlines()
    y_offset = box_y + 60
    for line in lines:
        line_surf = font_small.render(line, True, (0,0,0))
        screen.blit(line_surf, (box_x+20, y_offset))
        y_offset += line_surf.get_height() + 2

    # Two buttons: "Copy PGN" and "Close"
    button_width = 130
    button_height = 40
    spacing = 20
    total_width = 2*button_width + spacing
    start_x = box_x + (box_width - total_width)//2
    btn_y = box_y + box_height - button_height - 20

    copy_rect = pygame.Rect(start_x, btn_y, button_width, button_height)
    close_rect = pygame.Rect(start_x+button_width+spacing, btn_y, button_width, button_height)

    pygame.draw.rect(screen, (100,100,100), copy_rect)
    copy_text = font_large.render("Copy PGN", True, (255,255,255))
    screen.blit(copy_text, copy_text.get_rect(center=copy_rect.center))

    pygame.draw.rect(screen, (100,100,100), close_rect)
    close_text = font_large.render("Close", True, (255,255,255))
    screen.blit(close_text, close_text.get_rect(center=close_rect.center))

    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if copy_rect.collidepoint(event.pos):
                    pyperclip.copy(pgn_text)
                    print("PGN copied to clipboard.")
                elif close_rect.collidepoint(event.pos):
                    waiting = False
        pygame.time.wait(100)

# ---------------------------
# END-GAME DIALOGUE
# ---------------------------
def end_game_dialogue(screen, board, game_result, eval_history):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0,0,0,180))
    screen.blit(overlay, (0,0))

    box_width = 700
    box_height = 500
    box_x = (WIDTH - box_width) // 2
    box_y = (HEIGHT - box_height) // 2
    dialog_rect = pygame.Rect(box_x, box_y, box_width, box_height)
    pygame.draw.rect(screen, (240,240,240), dialog_rect)
    pygame.draw.rect(screen, (0,0,0), dialog_rect, 2)

    font_large = pygame.font.Font(None, 36)
    result_text = font_large.render(game_result, True, (0,0,0))
    result_rect = result_text.get_rect(center=(box_x + box_width//2, box_y + 30))
    screen.blit(result_text, result_rect)

    # Generate evaluation graph
    if eval_history:
        moves = list(range(1, len(eval_history)+1))
        plt.figure(figsize=(6.6, 3.2))
        plt.plot(moves, eval_history, marker='o')
        plt.title("Evaluation vs Move Number")
        plt.xlabel("Move Number")
        plt.ylabel("Evaluation (cp)")
        plt.axhline(y=0, color='gray', linestyle='--')
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='PNG')
        buf.seek(0)
        try:
            graph_surface = pygame.image.load(buf)
        except Exception as e:
            print("Error loading graph:", e)
            graph_surface = pygame.Surface((660,320))
            graph_surface.fill((200,200,200))
        plt.close()
        buf.close()
    else:
        graph_surface = pygame.Surface((660,320))
        graph_surface.fill((200,200,200))
        f = pygame.font.Font(None, 24)
        txt = f.render("No eval data", True, (0,0,0))
        graph_surface.blit(txt, txt.get_rect(center=(330,160)))
    
    graph_rect = pygame.Rect(box_x + 20, box_y + 60, 660, 320)
    screen.blit(graph_surface, graph_rect)

    # Define two buttons: Export PGN and Quit
    buttons = []
    button_width = 140
    button_height = 40
    spacing = 20
    total_width = 2 * button_width + spacing
    start_x = box_x + (box_width - total_width) // 2
    btn_y = box_y + box_height - button_height - 20

    export_rect = pygame.Rect(start_x, btn_y, button_width, button_height)
    quit_rect = pygame.Rect(start_x + button_width + spacing, btn_y, button_width, button_height)
    buttons.append((export_rect, "Export PGN"))
    buttons.append((quit_rect, "Quit"))

    font_button = pygame.font.Font(None, 28)
    for rect, label in buttons:
        pygame.draw.rect(screen, (100,100,100), rect)
        txt = font_button.render(label, True, (255,255,255))
        screen.blit(txt, txt.get_rect(center=rect.center))

    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                for rect, label in buttons:
                    if rect.collidepoint(pos):
                        if label == "Export PGN":
                            show_pgn_dialog(screen, board)
                            # After user closes the PGN dialog, we re-draw this end-game dialogue
                            pygame.display.flip()
                        elif label == "Quit":
                            waiting = False
        pygame.time.wait(100)

# ---------------------------
# MENU FUNCTIONS
# ---------------------------
def show_menu(screen):
    font = pygame.font.Font(None, 48)
    options = [
        ("Over the Board", None),
        ("Play vs Computer", "computer"),
        ("Play vs LLM", "llm")
    ]
    texts = [font.render(o[0], True, (0,0,0)) for o in options]
    rects = [txt.get_rect(center=(WIDTH//2, HEIGHT//2 + i*60 - 60)) for i, txt in enumerate(texts)]
    
    while True:
        screen.fill((200,200,200))
        for txt, rct in zip(texts, rects):
            screen.blit(txt, rct)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                for i, rct in enumerate(rects):
                    if rct.collidepoint(pos):
                        mode = options[i][1]
                        if mode is None:
                            return ("otb", None)
                        else:
                            return show_color_menu(screen, mode)
        pygame.time.wait(10)

def show_color_menu(screen, mode):
    font = pygame.font.Font(None, 48)
    options = [
        ("Play as White", chess.BLACK),  # Opponent takes Black
        ("Play as Black", chess.WHITE)   # Opponent takes White
    ]
    texts = [font.render(o[0], True, (0,0,0)) for o in options]
    rects = [txt.get_rect(center=(WIDTH//2, HEIGHT//2 + i*80 - 40)) for i, txt in enumerate(texts)]
    
    while True:
        screen.fill((200,200,200))
        for txt, rct in zip(texts, rects):
            screen.blit(txt, rct)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                for i, rct in enumerate(rects):
                    if rct.collidepoint(pos):
                        return (mode, options[i][1])
        pygame.time.wait(10)

def show_promotion_menu(screen, piece_color):
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0,0,0,150))
    screen.blit(overlay, (0,0))
    choices = [("q", chess.QUEEN), ("r", chess.ROOK), ("b", chess.BISHOP), ("n", chess.KNIGHT)]
    popup_width = SQUARE_SIZE * 4 + 40
    popup_height = SQUARE_SIZE + 40
    popup_x = (WIDTH - popup_width) // 2
    popup_y = (HEIGHT - popup_height) // 2
    rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)
    pygame.draw.rect(screen, (240,240,240), rect)
    pygame.draw.rect(screen, (0,0,0), rect, 2)
    buttons = []
    for i, (p_letter, p_const) in enumerate(choices):
        key = f"{'w' if piece_color==chess.WHITE else 'b'}{p_letter}"
        img = pygame.transform.scale(PIECES[key], (SQUARE_SIZE, SQUARE_SIZE))
        btn_x = popup_x + 10 + i*(SQUARE_SIZE+10)
        btn_y = popup_y + 20
        btn_rect = pygame.Rect(btn_x, btn_y, SQUARE_SIZE, SQUARE_SIZE)
        screen.blit(img, (btn_x, btn_y))
        pygame.draw.rect(screen, (0,0,0), btn_rect, 2)
        buttons.append((btn_rect, p_const))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                for btn_rect, p_const in buttons:
                    if btn_rect.collidepoint(event.pos):
                        return p_const
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.time.wait(10)

# ---------------------------
# MAIN GAME LOOP
# ---------------------------
def main():
    global current_fen, best_move_arrow, threat_move_arrow, last_arrow_update, updating_arrows
    global show_eval, show_best_arrow, show_threat_arrow, difficulty, play_mode, computer_color
    global history_scroll, eval_history, game_over_shown

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Chess GUI")

    stockfish_proc = subprocess.Popen(
        [STOCKFISH_PATH],
        universal_newlines=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    eval_thread = threading.Thread(target=stockfish_evaluation_loop, args=(stockfish_proc,), daemon=True)
    eval_thread.start()

    play_mode, computer_color = show_menu(screen)
    human_color = (not computer_color) if play_mode in ["computer", "llm"] else None

    board = chess.Board()
    current_fen = board.fen()
    clock = pygame.time.Clock()
    running = True
    selected_square = None
    flipped = False

    history_boards = [board.copy()]
    history_index = 0

    best_move_arrow = None
    threat_move_arrow = None
    last_arrow_update = 0
    updating_arrows = False

    ai_thinking = False

    while running:
        screen.fill((0,0,0))
        if show_eval:
            draw_evaluation_bar(screen)
        draw_board(screen, flipped)
        draw_pieces(screen, board, flipped)
        # This is the function that was missing:
        draw_captured_panels(screen, board)
        draw_last_move_highlight(screen, board, flipped)
        draw_selected_highlight(screen, board, selected_square, flipped)
        if show_best_arrow and best_move_arrow is not None:
            draw_move_arrow(screen, best_move_arrow, flipped, GREEN)
        if show_threat_arrow and threat_move_arrow is not None:
            draw_move_arrow(screen, threat_move_arrow, flipped, RED)
        draw_side_panel(screen)
        draw_move_history_panel(screen, board, history_scroll)
        pygame.display.flip()

        if board.is_game_over() and not game_over_shown:
            if board.is_checkmate():
                result = "Checkmate!"
            elif board.is_stalemate():
                result = "Stalemate!"
            else:
                result = "Game Over"
            game_over_shown = True
            end_game_dialogue(screen, board, result, eval_history)
            running = False
            continue

        if history_index == len(history_boards) - 1 and board.turn == computer_color:
            if not ai_thinking:
                ai_thinking = True
                def ai_move_thread():
                    nonlocal ai_thinking, board, history_boards, history_index
                    global current_fen
                    if play_mode == "computer":
                        mv = get_suggestion_for_color(board, computer_color, apply_difficulty=True)
                    else:
                        mv = get_llm_move(board, computer_color)
                    if mv in board.legal_moves:
                        board.push(mv)
                        history_boards.append(board.copy())
                        history_index += 1
                        current_fen = board.fen()
                        eval_history.append(eval_score)
                    ai_thinking = False
                t = threading.Thread(target=ai_move_thread)
                t.daemon = True
                t.start()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and play_mode in ["computer", "llm"]:
                    end_game_dialogue(screen, board, "Resignation", eval_history)
                    running = False
                    break
                elif play_mode == "otb":
                    if event.key == pygame.K_w:
                        end_game_dialogue(screen, board, "Resignation (White)", eval_history)
                        running = False
                        break
                    elif event.key == pygame.K_b:
                        end_game_dialogue(screen, board, "Resignation (Black)", eval_history)
                        running = False
                        break
                    elif event.key == pygame.K_d:
                        end_game_dialogue(screen, board, "Draw", eval_history)
                        running = False
                        break

                if event.key == pygame.K_BACKSPACE:
                    if play_mode in ["computer", "llm"]:
                        if len(board.move_stack) >= 2:
                            board.pop()
                            history_boards.pop()
                            board.pop()
                            history_boards.pop()
                        elif board.move_stack:
                            board.pop()
                            history_boards.pop()
                    else:
                        if board.move_stack:
                            board.pop()
                            history_boards.pop()
                    selected_square = None
                    history_index = len(history_boards) - 1
                    current_fen = board.fen()

                elif event.key == pygame.K_f:
                    flipped = not flipped

                elif event.key == pygame.K_LEFT and history_index > 0:
                    history_index -= 1
                    board = history_boards[history_index].copy()
                    current_fen = board.fen()

                elif event.key == pygame.K_RIGHT and history_index < len(history_boards) - 1:
                    history_index += 1
                    board = history_boards[history_index].copy()
                    current_fen = board.fen()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    history_scroll = max(history_scroll - 20, 0)
                elif event.button == 5:
                    font_small = pygame.font.Font(None, 20)
                    temp_board = chess.Board()
                    mv_list = [temp_board.san(mv) for mv in board.move_stack]
                    for mv in board.move_stack:
                        temp_board.push(mv)
                    lines = [f"{(i//2)+1}. {mv_list[i]}" + (f" {mv_list[i+1]}" if i+1 < len(mv_list) else "")
                             for i in range(0, len(mv_list), 2)]
                    line_height = font_small.get_height() + 4
                    total_text_height = line_height * len(lines)
                    max_scroll = max(total_text_height - (HISTORY_PANEL_HEIGHT - 10), 0)
                    history_scroll = min(history_scroll + 20, max_scroll)
                else:
                    pos = event.pos
                    if pos[0] >= EVAL_BAR_WIDTH + BOARD_SIZE:
                        if FLIP_BUTTON_RECT.collidepoint(pos):
                            flipped = not flipped
                        elif PREV_BUTTON_RECT.collidepoint(pos) and history_index > 0:
                            history_index -= 1
                            board = history_boards[history_index].copy()
                            current_fen = board.fen()
                        elif NEXT_BUTTON_RECT.collidepoint(pos) and history_index < len(history_boards) - 1:
                            history_index += 1
                            board = history_boards[history_index].copy()
                            current_fen = board.fen()
                        elif TAKEBACK_BUTTON_RECT.collidepoint(pos):
                            if play_mode in ["computer", "llm"]:
                                if len(board.move_stack) >= 2:
                                    board.pop()
                                    history_boards.pop()
                                    board.pop()
                                    history_boards.pop()
                                elif board.move_stack:
                                    board.pop()
                                    history_boards.pop()
                            else:
                                if board.move_stack:
                                    board.pop()
                                    history_boards.pop()
                            selected_square = None
                            history_index = len(history_boards) - 1
                            current_fen = board.fen()
                        elif EVAL_TOGGLE_BUTTON_RECT.collidepoint(pos):
                            show_eval = not show_eval
                        elif BEST_ARROW_TOGGLE_BUTTON_RECT.collidepoint(pos):
                            show_best_arrow = not show_best_arrow
                        elif THREAT_ARROW_TOGGLE_BUTTON_RECT.collidepoint(pos):
                            show_threat_arrow = not show_threat_arrow
                        elif DIFF_UP_BUTTON_RECT.collidepoint(pos):
                            difficulty = min(difficulty + 100, 3200)
                        elif DIFF_DOWN_BUTTON_RECT.collidepoint(pos):
                            difficulty = max(difficulty - 100, 100)
                    else:
                        if history_index == len(history_boards) - 1:
                            col = (pos[0] - EVAL_BAR_WIDTH) // SQUARE_SIZE
                            row = (pos[1] - TOP_PANEL_HEIGHT) // SQUARE_SIZE
                            sq = chess.square(7 - col, row) if flipped else chess.square(col, 7 - row)
                            if selected_square is None:
                                selected_square = sq
                            else:
                                piece = board.piece_at(selected_square)
                                mv = chess.Move(selected_square, sq)
                                if piece and piece.symbol().lower() == 'p':
                                    rank = chess.square_rank(sq)
                                    if (piece.color == chess.WHITE and rank == 7) or (piece.color == chess.BLACK and rank == 0):
                                        prom = show_promotion_menu(screen, piece.color)
                                        mv = chess.Move(selected_square, sq, promotion=prom)
                                if mv in board.legal_moves:
                                    board.push(mv)
                                    history_boards.append(board.copy())
                                    history_index += 1
                                    current_fen = board.fen()
                                    eval_history.append(eval_score)
                                selected_square = None

        if (time.time() - last_arrow_update > arrow_update_interval) and not updating_arrows:
            updating_arrows = True
            board_copy = board.copy()
            t = threading.Thread(target=update_arrows_thread, args=(board_copy, play_mode, computer_color, flipped))
            t.daemon = True
            t.start()

        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
