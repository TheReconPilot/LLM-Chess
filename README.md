# Chess LLM

This is a joint project by [Purva Parmar](https://thereconpilot.github.io/) and [Ayan Biswas](https://www.linkedin.com/in/ayan-biswas-180b07159/) for the course [E2 335 - Topics in Artificial Intelligence 2025](https://sway.cloud.microsoft/11WDvDi9GKVb7e7r) taught by [Prof. Aditya Gopalan](https://ece.iisc.ac.in/~aditya/) at the Indian Institute of Science, Bengaluru for the January 2025 Semester.

## Introduction

We explore the capabilities of Large Language Models (LLMs) in playing chess. We test various open-source LLMs with different parameter sizes (1B-8B) against the Stockfish chess engine to analyze their performance. We also do some Supervised Fine-Tuning, which only modestly improves the performance.

## Project Layout

The original project folder was quite messy, with various chunks of codes repeated in various notebooks and scripts for experimentation. This repo is a somewhat cleaned up version where we compile most things.

The primary files to take note of are `llmchess_experiments.py`, `sft.py`, `analysis.py`, the DatasetCreation Notebooks and the interactive game inside the `Interactive/` directory.


**List of files**
- `.env-example` - The environment variables file (loaded with python-dotenv). Rename the file to `.env` and replace the placeholders inside to their appropriate values.
- `llmchess_experiments.py` - Runs chess games of LLM vs Stockfish. The config at the end handles most stuff. Take note of the `fine_tuned` parameter, which if set to True, attempts to load adapters stored from fine tuning models by the `sft.py` script.
- `attack_threat_prompt.py` and `cot_pattern_prompt.py` - Generates some different prompt styles which are called in `llmchess_experiments.py`
- `DatasetCreation.ipynb` and `DatasetCreation-AttackThreatMap.ipynb` - Generate dataset for a couple of different prompt styles, for Supervised Fine-Tuning.
- `sft.py` - Supervised Fine-Tuning for the models with Quantization, PEFT and LORA. Adapters stored in mentioned directory.
- `run_sft_jobs.py` - Extra script we used to schedule the `sft.py` across a setup of 3 GPUs.
- `schedule_sft_experiments.py` - Extra script we used to schedule experiments in `llmchess_experiments.py` across a setup fo 3 GPUs.
- `analysis.py` - Analysis code for generated `.jsonl` result files.
- `Interactive/interactive_chess.py` - Contains an interactive pygame based game to play chess against Stockfish or LLMs.

## Acknowledgements

We thank Prof. Aditya Gopalan for the course and the guidance he provided throughout the project.

**Resources**
- 1x Nvidia A4000 16GB GPU at TATA ELXSI AI Lab at ECE Department, IISc
- 3x Nvidia V100 32 GB GPUs at the Arjuna ECE Cluster at ECE Department, IISc

## Contact

- Purva Parmar - purvaparmar@iisc.ac.in
- Ayan Biswas - ayanbiswas@iisc.ac.in
