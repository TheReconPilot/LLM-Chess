import os
import chess
import chess.engine
import re
import random
import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import gc
from peft import PeftModel

from attack_threat_prompt import build_atm_prompt
from cot_pattern_prompt import build_cot_pattern_prompt

from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(os.environ['HF_TOKEN'])

@dataclass
class GameMetrics:
    model_name: str
    prompt_method: str
    parameters: dict
    result: str  # "win", "loss", "draw"
    move_accuracy: float  # percentage of moves in Stockfish top n
    avg_move_quality: float
    game_duration: float  # seconds
    num_legal_moves_suggested: int
    total_moves: int
    board_evaluations: List[float]
    moves_played: List[str]
    stockfish_top_moves: List[List[str]]
    llm_suggested_moves: List[str]
    timestamp: str

class LLMChessPlayer:
    def __init__(self, model_name: str, device: str = "cuda", quantized: bool = True, fine_tuned: bool = False, adapter_dir: str = None):
        self.base_model_name = model_name
        self.adapter_name = model_name.split("/")[-1]
        # Adjust this adapter path
        if adapter_dir:
            self.adapter_path = os.path.join(adapter_dir, self.adapter_name)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if quantized:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

        if fine_tuned:
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)

    def generate_move(self, prompt: str, parameters: dict) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=parameters.get("max_new_tokens", 100),
                temperature=parameters.get("temperature", 0.7),
                top_p=parameters.get("top_p", 0.9),
                do_sample=parameters.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id
            )
        # Get only the new tokens, not including the prompt
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

class ChessExperiment:
    def __init__(self, stockfish_path: str, stockfish_level: int = 5):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine.configure({"Skill Level": stockfish_level})
        self.board = chess.Board()

    def reset_game(self):
        self.board.reset()

    def get_legal_moves(self) -> List[str]:
        return [self.board.san(move) for move in self.board.legal_moves]

    def get_board_evaluation(self, board: Optional[chess.Board] = None) -> float:
        board = board or self.board
        info = self.engine.analyse(board, chess.engine.Limit(depth=10))
        score = info["score"].white()
        if score.is_mate():
            return 1000.0 if score.mate() > 0 else -1000.0
        return score.score() / 100.0

    def get_stockfish_top_moves(self, n: int = 3) -> List[str]:
        info = self.engine.analyse(self.board, chess.engine.Limit(depth=10), multipv=n)
        return [self.board.san(move["pv"][0]) for move in info]

    def get_stockfish_top_move_eval(self, n: int = 1) -> Tuple[str, float]:
        info = self.engine.analyse(self.board, chess.engine.Limit(depth=10), multipv=n)
        best = info[0]
        move = self.board.san(best["pv"][0])
        score = best["score"].white()
        eval_score = 1000.0 if score.is_mate() and score.mate() > 0 else (-1000.0 if score.is_mate() else score.score() / 100.0)
        return move, eval_score

    def play_stockfish_move(self) -> Tuple[str, float]:
        result = self.engine.play(self.board, chess.engine.Limit(time=0.1))
        move = self.board.san(result.move)
        self.board.push(result.move)
        eval_after = self.get_board_evaluation()
        return move, eval_after

    def play_llm_move(self, llm_player: LLMChessPlayer, prompt_method: str, parameters: dict) -> Tuple[str, float, List[str], str]:
        legal_moves = self.get_legal_moves()

        if prompt_method == "state_only":
            prompt = f"Current chess position (FEN): {self.board.fen()}\nWhat is the best move in this position? Respond with just the move in standard algebraic notation."
        elif prompt_method == "legal_moves":
            prompt = f"Current chess position (FEN): {self.board.fen()}\nLegal moves: {', '.join(legal_moves)}\nWhat is the best move from these options? Respond with just the move."
        elif prompt_method == "chain_of_thought":
            prompt = f"""Analyze this chess position:

                {self.board.fen()}

                Instructions:
                1. Analyze the position
                2. Consider material and king safety
                3. Choose the best move

                Format your response as:
                THOUGHT: <your analysis>
                MOVE: <selected move in SAN format> (e.g., e2e4, g7g8q, O-O, O-O-O)"""
        elif prompt_method == "attack_threat_map":
            prompt = build_atm_prompt(self.board.copy())
        elif prompt_method == "cot_pattern":
            prompt = build_cot_pattern_prompt(self.board.copy())
        elif prompt_method == "3_step_prompt":
            pass  # To be handled separately
        else:
            raise ValueError("Invalid prompt method")

        if prompt_method == "3_step_prompt":
            legal_moves = self.get_legal_moves()

            # Step 1: Ask for legal moves
            prompt_legal_moves = f"Current chess position (FEN): {self.board.fen()}\nWhat are the legal moves in this position? Respond with a comma-separated list of moves."
            response_legal_moves = llm_player.generate_move(prompt_legal_moves, parameters)
            llm_legal_moves = [move.strip() for move in response_legal_moves.split(",") if move.strip() in legal_moves]
            
            print(f"{'[ Step 1 Prompt (3 Step Prompt Method) ]':=^100}")
            print(prompt_legal_moves)

            print(f"{'[ Step 1 Response (3 Step Prompt Method) ]':=^100}")
            print(response_legal_moves)

            # Step 2: Ask for possible strategies
            prompt_strategies = f"Given the legal moves: {', '.join(llm_legal_moves)}\nWhat are the possible strategies in this position? Respond with a brief description of each strategy."
            response_strategies = llm_player.generate_move(prompt_strategies, parameters)

            print(f"{'[ Step 2 Prompt (3 Step Prompt Method) ]':=^100}")
            print(prompt_strategies)

            print(f"{'[ Step 2 Response (3 Step Prompt Method) ]':=^100}")
            print(response_strategies)

            # Step 3: Ask for the best move
            prompt_best_move = f"Based on the strategies: {response_strategies}\nWhat is the best move in this position? Respond with just the move in standard algebraic notation."
            response_best_move = llm_player.generate_move(prompt_best_move, parameters)

            response = response_best_move  # For proper return at the end

            # Extract the move from the response
            suggested_moves = self._extract_moves_from_response(response_best_move, prompt_method)
            legal_suggestions = [move for move in suggested_moves if move in legal_moves]

            move_str = legal_suggestions[0] if legal_suggestions else random.choice(legal_moves)
            
            print(f"{'[ Step 3 Prompt (3 Step Prompt Method) ]':=^100}")
            print(prompt_best_move)

            print(f"{'[ Step 3 Response (3 Step Prompt Method) ]':=^100}")
            print(response_best_move)

            print(f"{'[ Suggested Move) ]':=^100}")
            print(move_str)

        else:  # Other prompt methods
            print(f"{f'[ Prompt ({prompt_method}) ]':=^100}")
            print(prompt)
            response = llm_player.generate_move(prompt, parameters)
            print(f"{'[ Response ]':=^100}")
            print(response)
            suggested_moves = self._extract_moves_from_response(response, prompt_method)
            legal_suggestions = [move for move in suggested_moves if move in legal_moves]

            move_str = legal_suggestions[0] if legal_suggestions else random.choice(legal_moves)
            print(f"{'[ Suggested Move ]':=^100}")
            print(move_str)
        
        # Parse move
        try:
            move = self.board.parse_san(move_str)
        except ValueError:
            try:
                move = self.board.parse_uci(move_str.lower())
            except ValueError:
                move = self.board.parse_san(random.choice(legal_moves))

        print(f"{'[ Played Move ]':=^100}")
        print(move)

        self.board.push(move)
        eval_after = self.get_board_evaluation()

        return move_str, eval_after, legal_suggestions, response

    def _extract_moves_from_response(self, response: str, prompt_method: str) -> List[str]:
        if prompt_method in ["chain_of_thought", "attack_threat_map", "cot_pattern"]:
            move_pattern = r"(?i)\bmove:\s*([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O|O-O-O)\b"
        else:
            move_pattern = r"\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O|O-O-O)\b"
        matches = re.findall(move_pattern, response)
        return [m.strip() for m in matches]

    def run_game(self, llm_player: LLMChessPlayer, prompt_method: str, parameters: dict, top_n_accuracy: int = 3) -> GameMetrics:
        self.reset_game()
        metrics = {
            "board_evaluations": [self.get_board_evaluation()],
            "moves_played": [],
            "stockfish_top_moves": [],
            "llm_suggested_moves": [],
            "num_legal_moves_suggested": 0,
            "accurate_moves": 0,
            "move_quality_diffs": [],
            "start_time": time.time()
        }

        game_over = False
        while not game_over and self.board.fullmove_number < 200:
            if not self.board.is_game_over():
                move, eval_after, legal_suggestions, llm_response = self.play_llm_move(llm_player, prompt_method, parameters)
                stockfish_top_moves = self.get_stockfish_top_moves(top_n_accuracy)
                best_sf_move, best_sf_eval = self.get_stockfish_top_move_eval(1)

                metrics["moves_played"].append(move)
                metrics["board_evaluations"].append(eval_after)
                metrics["stockfish_top_moves"].append(stockfish_top_moves)
                metrics["llm_suggested_moves"].append(llm_response)
                metrics["num_legal_moves_suggested"] += min(1, len(legal_suggestions))

                if move in stockfish_top_moves:
                    metrics["accurate_moves"] += 1

                metrics["move_quality_diffs"].append(best_sf_eval - eval_after)

            if not self.board.is_game_over():
                move, eval_after = self.play_stockfish_move()
                metrics["moves_played"].append(f"sf:{move}")
                metrics["board_evaluations"].append(eval_after)

            game_over = self.board.is_game_over()

        result = "draw"
        if self.board.is_checkmate():
            result = "loss" if self.board.turn == chess.WHITE else "win"


        total_moves = len([m for m in metrics["moves_played"] if not m.startswith("sf:")])
        move_accuracy = metrics["accurate_moves"] / total_moves if total_moves > 0 else 0
        avg_move_quality = sum(metrics["move_quality_diffs"]) / len(metrics["move_quality_diffs"]) if metrics["move_quality_diffs"] else 0

        return GameMetrics(
            model_name=llm_player.base_model_name,
            prompt_method=prompt_method,
            parameters=parameters,
            result=result,
            move_accuracy=move_accuracy,
            avg_move_quality=avg_move_quality,
            game_duration=time.time() - metrics["start_time"],
            num_legal_moves_suggested=metrics["num_legal_moves_suggested"],
            total_moves=total_moves,
            board_evaluations=metrics["board_evaluations"],
            moves_played=metrics["moves_played"],
            stockfish_top_moves=metrics["stockfish_top_moves"],
            llm_suggested_moves=metrics["llm_suggested_moves"],
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def close(self):
        self.engine.quit()

def save_metrics(metrics: GameMetrics, filename: str):
    with open(filename, "a") as f:
        f.write(json.dumps(asdict(metrics)) + "\n")

def run_experiments(
    stockfish_path: str,
    model_names: List[str],
    prompt_methods: List[str],
    parameters_list: List[dict],
    num_games: int,
    output_file: str,
    fine_tuned: bool = False,
    adapter_dir: str = None
):
    # Heuristics for deciding if a model should be quantized
    quantized_models = {"mistral", "meta-llama", "gemma", "openchat"}

    for model_name in model_names:
        use_quantized = any(q in model_name.lower() for q in quantized_models)
        print("="*150)
        print(f"\nLoading model: {model_name} (quantized={use_quantized})")

        llm_player = LLMChessPlayer(model_name, quantized=use_quantized, fine_tuned=fine_tuned, adapter_dir=adapter_dir)

        for prompt_method in prompt_methods:
            print(f"\nPrompt Method: {prompt_method}")
            for parameters in parameters_list:
                print(f"Experimenting with {prompt_method = } and {parameters = }")
                experiment = ChessExperiment(stockfish_path)

                for gameidx in range(num_games):
                    metrics = experiment.run_game(llm_player, prompt_method, parameters)
                    save_metrics(metrics, output_file)
                    print(f"Completed Game {gameidx + 1}/{num_games}. Game duration = {metrics.game_duration} seconds")

                experiment.close()
                del experiment

        # Explicit cleanup
        del llm_player.model
        del llm_player.tokenizer
        del llm_player
        torch.cuda.empty_cache()
        gc.collect()



# Example usage
if __name__ == "__main__":
    load_dotenv()
    login(os.environ['HF_TOKEN'])

    CONFIG = {
        "stockfish_path": os.environ['stockfish_path'],
        "model_names": [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "openchat/openchat-3.5-0106",
            "google/gemma-7b-it",
            "deepseek-ai/deepseek-coder-1.3b-instruct",
            "microsoft/phi-2",
            "microsoft/Phi-4-mini-instruct",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ],
        "prompt_methods": [
            "state_only", 
            "legal_moves", 
            "chain_of_thought", 
            "attack_threat_map", 
            "cot_pattern",
            "3_step_prompt"
        ],
        "parameters_list": [
            {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 128},
            # {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 50},
        ],
        "num_games": 10,
        "output_file": "chess_metrics_sft.jsonl",
        "fine_tuned": False,    # Are we running experiments on fine-tuned models? If yes, also need to pass adapter_path below
        "adapter_dir": "chess-sft-outputs/"   # Only needed if running experiments on fine-tuned models
    }

    run_experiments(**CONFIG)
