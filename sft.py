#!/usr/bin/env python3
import os
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, Dataset, load_from_disk
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainerCallback
)
import peft
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import wandb
import huggingface_hub
from dotenv import load_dotenv

def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT on chess dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Model identifier from HuggingFace")
    parser.add_argument("--dataset_path", type=str, default="chess-sft-dataset", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./chess-sft-outputs", help="Output directory")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="Logging directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--grad_accum_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA r")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--wandb_project", type=str, default="chess-sft", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    return parser.parse_args()

def load_and_prepare_dataset(dataset_path, tokenizer, max_seq_length, model_name):
    if os.path.isdir(dataset_path):
        dataset = load_from_disk(dataset_path)
    else:
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = Dataset.from_list(data)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    def tokenize_function(examples):
        if "mistral" in model_name.lower() or "llama" in model_name.lower():
            text = f"[INST] {examples['prompt']} [/INST] {examples['response']}"
        elif "phi" in model_name.lower():
            text = f"Instruct: {examples['prompt']}\nOutput: {examples['response']}"
        elif "gemma" in model_name.lower():
            text = f"<start_of_turn>user\n{examples['prompt']}<end_of_turn>\n<start_of_turn>model\n{examples['response']}<end_of_turn>"
        elif "openchat" in model_name.lower():
            text = f"GPT4 User: {examples['prompt']}<|end_of_turn|>GPT4 Assistant: {examples['response']}<|end_of_turn|>"
        else:
            text = f"User: {examples['prompt']}\nAssistant: {examples['response']}"
        
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        prompt_len = len(tokenizer(examples['prompt'], truncation=True, max_length=max_seq_length)['input_ids'])
        tokenized["labels"] = tokenized["input_ids"].clone()
        tokenized["labels"][:, :prompt_len] = -100
        
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset"
    )
    
    return tokenized_dataset


def load_model(model_name, device):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,  
        bnb_8bit_use_double_quant=True,  
        bnb_8bit_quant_type="nf4",
        bnb_8bit_compute_dtype=torch.bfloat16
    )
    
    needs_quantization = any(m in model_name.lower() for m in [
        "mistral",  
        "meta-llama", 
        "gemma", 
        "openchat"
    ])
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config if needs_quantization else None,
        device_map=f"cuda:{device}",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    if needs_quantization:
        model = prepare_model_for_kbit_training(model)
    
    return model

def create_peft_config(model, model_name, lora_r, lora_alpha, lora_dropout):
    if "phi-2" in model_name.lower():
        target_modules = ["query_key_value", "dense"]
    elif "phi-4" in model_name.lower():
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    elif "gemma" in model_name.lower():
        target_modules = ["q_proj", "v_proj"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    return peft_config

def main():
    load_dotenv()
    huggingface_hub.login(os.environ['HF_TOKEN'])
    args = parse_args()
    
    # Initialize W&B
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.model_name.split('/')[-1],
        config=vars(args)
    )
    
    # Create output directories
    output_dir = os.path.join(args.output_dir, args.model_name.split('/')[-1])
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare dataset
    tokenized_dataset = load_and_prepare_dataset(
        args.dataset_path, 
        tokenizer, 
        args.max_seq_length, 
        args.model_name
    )
    
    # Load model
    model = load_model(args.model_name, args.device)
    
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()

    model.gradient_checkpointing_enable()

    # Set up LoRA
    peft_config = create_peft_config(
        model, 
        args.model_name, 
        args.lora_r, 
        args.lora_alpha, 
        args.lora_dropout
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Reduce batch size for gemma model to fit in memory
    batch_size = 2 if "gemma" in args.model_name.lower() else args.batch_size
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=3000,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_dir=args.logging_dir,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        # fp16=torch.cuda.is_available(),
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # model.train()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )
    
    # Train
    print(f"Starting training for {args.model_name} on device cuda:{args.device}")
    torch.cuda.empty_cache()
    # Add code to periodically clear cache during training
    class GarbageCollectionCallback(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kwargs):
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    trainer.add_callback(GarbageCollectionCallback())
    trainer.train()
    
    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    wandb.finish()
    print(f"Training complete for {args.model_name}!")

if __name__ == "__main__":
    load_dotenv()
    huggingface_hub.login(os.environ['HF_TOKEN'])
    wandb.login()
    main()
