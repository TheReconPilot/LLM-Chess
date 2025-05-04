from typing import Dict
from collections import defaultdict
import chess


def get_threat_map(board: chess.Board) -> Dict[str, Dict[str, str]]:
    """
    Generates a threat map highlighting which white pieces are under threat from black pieces.
    Includes one-move checkmate threats.
    """
    threat_map = defaultdict(dict)
    board1=board.copy()
    # Iterate over all squares
    for square in chess.SQUARES:
        square_name = chess.square_name(square)

        # Check if the square is under attack by black
        attackers = board1.attackers(chess.BLACK, square)
        if attackers:
            piece = board1.piece_at(square)
            if piece and piece.color == chess.WHITE:  # Only consider white pieces
                threat_map[square_name] = {
                    "attackers": [chess.square_name(attacker) for attacker in attackers],
                    "piece": piece.symbol().upper()
                }

    # Check for one-move checkmate threats
    for move in board1.legal_moves:
        if board1.turn == chess.WHITE:  # Only consider black's turn
            board1.push(move)
            if board1.is_checkmate():
                king_square = chess.square_name(board1.king(chess.WHITE))
                threat_map["Checkmate Threat"] = {
                    "move": move.uci(),
                    "target_king": king_square
                }
            board1.pop()

    return threat_map
def get_attack_map(board: chess.Board) -> Dict[str, Dict[str, str]]:
    """
    Generates a threat map highlighting which black pieces are under threat from white pieces.
    Includes one-move checkmate threats.
    """
    threat_map = defaultdict(dict)
    board1 = board.copy()

    # Iterate over all squares
    for square in chess.SQUARES:
        square_name = chess.square_name(square)

        # Check if the square is under attack by white
        attackers = board1.attackers(chess.WHITE, square)
        if attackers:
            piece = board1.piece_at(square)
            if piece and piece.color == chess.BLACK:  # Only consider black pieces
                threat_map[square_name] = {
                    "attackers": [chess.square_name(attacker) for attacker in attackers],
                    "piece": piece.symbol().upper()
                }

    # Check for one-move checkmate threats
    for move in board1.legal_moves:
        if board1.turn == chess.BLACK:  # Only consider white's turn
            board1.push(move)
            if board1.is_checkmate():
                king_square = chess.square_name(board1.king(chess.BLACK))
                threat_map["Checkmate Threat"] = {
                    "move": move.uci(),
                    "target_king": king_square
                }
            board1.pop()

    return threat_map


def build_atm_prompt(board: chess.Board) -> str:
    """
    Builds a well-engineered prompt for an LLM to analyze a chess position.
    The prompt includes the FEN string, attack map, threat map, and instructions
    to carefully assess threats and attacks before suggesting the best move.
    """
    # Get the attack and threat maps
    threat_map = get_threat_map(board)
    attack_map = get_attack_map(board)

    # Format the attack map
    attack_map_str = "\n".join(
        f"{square}: {details}" for square, details in attack_map.items()
    )

    # Format the threat map
    threat_map_str = "\n".join(
        f"{square}: {details}" for square, details in threat_map.items()
    )

    # Build the prompt
    prompt = (
        f"Analyze this chess position:\n\n"
        f"FEN:\n{board.fen()}\n\n"
        f"Attack Map:\n{attack_map_str}\n\n"
        f"Threat Map:\n{threat_map_str}\n\n"
        f"Instructions:\n"
        f"1. Carefully assess the threats from the Threat Map and the attacks from the Attack Map.\n"
        f"2. Consider material, king safety, and positional advantages.\n"
        f"3. Suggest the best move from the legal options.\n\n"
        f"Format your response as:\n"
        f"THOUGHT: <your analysis>\n"
        f"MOVE: <selected move>\n"
    )

    return prompt