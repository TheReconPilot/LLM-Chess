# Script to run experiments for SFT models across 3 GPUs.

import os
import time
import json
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import threading
import subprocess
from typing import List, Dict
from dotenv import load_dotenv
import huggingface_hub
import chess

load_dotenv()
huggingface_hub.login(os.environ['HF_TOKEN'])

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
            # "attack_threat_map", 
            # "cot_pattern",
            # "3_step_prompt"
        ],
    "parameters_list": [
        {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 256},
        # {"temperature": 0.3, "top_p": 0.9, "max_new_tokens": 100},
    ],
    "num_games": 1,
    "output_file": "chess_metrics_sft.jsonl",
    "fine_tuned": True,
    "adapter_dir": "chess-sft-outputs"
}

class ExperimentRunner:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.env = os.environ.copy()
        self.env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
    def run_experiment(self, model_name: str):
        load_dotenv()
        huggingface_hub.login(os.environ['HF_TOKEN'])
        print(f"Starting experiment for {model_name} on GPU {self.gpu_id}")
        start_time = time.time()
        
        # Construct the command to run the experiment as a separate process
        cmd = [
            "python3",
            "-c",
            f"""
import chess
from dotenv import load_dotenv
import huggingface_hub
import os
load_dotenv()
huggingface_hub.login(os.environ['HF_TOKEN'])
from llmchess_experiments import run_experiments
config = {{
    "stockfish_path": "{CONFIG['stockfish_path']}",
    "model_names": ["{model_name}"],
    "prompt_methods": {json.dumps(CONFIG['prompt_methods'])},
    "parameters_list": {json.dumps(CONFIG['parameters_list'])},
    "num_games": {CONFIG['num_games']},
    "output_file": "results/{CONFIG['output_file']}",
    "fine_tuned": {CONFIG['fine_tuned']},
    "adapter_dir": {CONFIG['adapter_dir']}
}}
run_experiments(**config)
            """
        ]
        
        try:
            # Run the experiment in a subprocess with the specified GPU
            result = subprocess.run(
                cmd,
                env=self.env,
                check=True,
                text=True,
                capture_output=True
            )
            print(f"Completed {model_name} on GPU {self.gpu_id} in {time.time()-start_time:.2f}s")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running {model_name} on GPU {self.gpu_id}:")
            print(e.stderr)
        
        return model_name

def gpu_worker(gpu_id: int, task_queue: Queue):
    runner = ExperimentRunner(gpu_id)
    while True:
        model_name = task_queue.get()
        if model_name is None:  # Sentinel value to stop the worker
            task_queue.task_done()
            break
            
        try:
            runner.run_experiment(model_name)
        finally:
            task_queue.task_done()

def schedule_experiments():
    # Create a queue and add all models to it
    task_queue = Queue()
    for model_name in CONFIG["model_names"]:
        task_queue.put(model_name)
    
    # Create and start worker threads (one per GPU)
    num_gpus = 3
    workers = []
    for gpu_id in range(num_gpus):
        worker = threading.Thread(
            target=gpu_worker,
            args=(gpu_id, task_queue)
        )
        worker.start()
        workers.append(worker)
        # Add a small delay between starting workers to avoid initialization conflicts
        time.sleep(5)
    
    # Block until all tasks are done
    task_queue.join()
    
    # Stop workers by sending None for each
    for _ in range(num_gpus):
        task_queue.put(None)
    for worker in workers:
        worker.join()

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    print("Starting experiment scheduler with 3 GPUs")
    schedule_experiments()
    print("All experiments completed!")
