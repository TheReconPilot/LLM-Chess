import chess
from typing import List, Dict
from collections import defaultdict

def expand_fen(fen: str) -> str:
    board_rep = []
    rows = fen.split()[0].split('/')
    for i, row in enumerate(rows):
        rank = 8 - i
        expanded = []
        for c in row:
            if c.isdigit():
                expanded.extend(['.'] * int(c))
            else:
                expanded.append(c)
        board_line = f"{rank} " + " ".join(expanded)
        board_rep.append(board_line)
    return "\n".join(board_rep) + "\n  a b c d e f g h"

def get_attack_map(board: chess.Board) -> str:
    attack_map = defaultdict(lambda: defaultdict(list))
    for square in chess.SQUARES:
        for color in [chess.WHITE, chess.BLACK]:
            attackers = board.attackers(color, square)
            for attacker in attackers:
                piece = board.piece_at(attacker)
                if piece:
                    color_name = "White" if color == chess.WHITE else "Black"
                    attack_map[color_name][chess.square_name(square)].append(piece.symbol().upper())
    return attack_map

def detect_patterns(board: chess.Board) -> Dict[str, List[str]]:
    patterns = defaultdict(list)
    
    # Check for checks
    if board.is_check():
        patterns["checks"] = ["Check!"]
    
    # Detect forks
    knights = board.pieces(chess.KNIGHT, board.turn)
    for knight in knights:
        attacks = board.attacks(knight)
        if len(attacks) >= 2:
            patterns["forks"].append(f"N{chess.square_name(knight)}")
    
    # Detect pins
    for square in chess.SQUARES:
        if board.is_pinned(board.turn, square):
            patterns["pins"].append(chess.square_name(square))
    
    return patterns

def _format_patterns(patterns: Dict[str, List[str]]) -> str:
    output = []
    for pattern, items in patterns.items():
        if items:
            output.append(f"- {pattern.capitalize()}: {', '.join(items[:3])}")
    return "\n".join(output) if output else "No major patterns detected"

def build_cot_pattern_prompt(board: chess.Board) -> str:
    fen = board.fen()
    patterns = detect_patterns(board)
    legal_moves = [board.san(move) for move in board.legal_moves]

    prompt = f"""Analyze this chess position:

            {expand_fen(fen)}

            Detected Patterns:
            {_format_patterns(patterns)}

            Legal Moves: {', '.join(legal_moves)}

            Instructions:
            1. Analyze the position using detected patterns
            2. Consider material and king safety
            3. Choose the best move from legal options

            Format your response as:
            THOUGHT: <your analysis>
            MOVE: <selected move in SAN format> (e.g., e2e4, g7g8q, O-O, O-O-O)"""
    return prompt