# Script to run SFT script on a setup with 3 GPUs.

import subprocess
import time
import os
from queue import Queue
import huggingface_hub
from dotenv import load_dotenv

load_dotenv()
huggingface_hub.login(os.environ['HF_TOKEN'])

models_to_train = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "openchat/openchat-3.5-0106",
    "google/gemma-7b-it",
    "deepseek-ai/deepseek-coder-1.3b-instruct",
    "microsoft/phi-2",
    "microsoft/Phi-4-mini-instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
]

# List of *physical* GPU IDs available on your machine
physical_gpus = [0, 1, 2]

# Track running jobs
running_jobs = []
model_queue = Queue()

for model in models_to_train:
    model_queue.put(model)

output_dir = "./chess-sft-outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

def launch_job(model_name, visible_gpu):
    print(f"Launching {model_name} on physical GPU {visible_gpu}")
    model_id_safe = model_name.replace("/", "_")
    log_file = f"logs/{model_id_safe}.log"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(visible_gpu)  # mask to 1 GPU
    load_dotenv()
    huggingface_hub.login(os.environ['HF_TOKEN'])
    cmd = [
        "python", "sft.py",
        "--model_name", model_name,
        "--wandb_run_name", f"{model_id_safe}_sft",
        "--output_dir", output_dir
    ]
    log = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
    return (proc, model_name, visible_gpu, log)

# Launch initial jobs
for gpu_id in physical_gpus:
    if not model_queue.empty():
        load_dotenv()
        huggingface_hub.login(os.environ['HF_TOKEN'])
        model = model_queue.get()
        time.sleep(2)
        running_jobs.append(launch_job(model, gpu_id))

# Monitor and launch new jobs as GPUs free up
while running_jobs:
    time.sleep(10)
    still_running = []
    for proc, model, gpu_id, log in running_jobs:
        if proc.poll() is None:
            still_running.append((proc, model, gpu_id, log))
        else:
            log.close()
            print(f"Job {model} on GPU {gpu_id} finished.")
            if not model_queue.empty():
                next_model = model_queue.get()
                still_running.append(launch_job(next_model, gpu_id))
    running_jobs = still_running

print("All jobs completed.")
