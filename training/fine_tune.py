"""
Chao AI Assistant - Fine-tuning Script
Uses Unsloth with Qwen3 model to create a personalized assistant
"""

import os
import json
import torch
import random
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from trl import SFTTrainer

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

ASSISTANT_NAME = config["assistant_name"]
OUTPUT_DIR = f"./models/{ASSISTANT_NAME.lower()}-assistant"

def main():
    set_seed(42)
    
    # Load Qwen3 model with Unsloth optimization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",  # You can choose other sizes: 0.6B, 1.7B, 4B, 8B, 14B
        max_seq_length=2048,  # Context length - can be increased up to 8x with Unsloth
        load_in_4bit=True,    # Quantization for memory efficiency
        token=None,           # Add your HF token if needed
    )
    
    # Set up LoRA tuning
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,                 # Rank of the update matrices
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"], 
        lora_alpha=32,        # Alpha parameter for LoRA scaling
        lora_dropout=0.05,    # Dropout probability for LoRA layers
        bias="none",          # Whether to train bias parameters
    )
    
    # Load and prepare dataset
    dataset = load_dataset("json", data_files="./data/chao_dataset.jsonl")
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )
    
    # Initialize the SFT trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        args=training_args,
        dataset_text_field="messages",
        max_seq_length=2048,
        packing=False,
    )
    
    # Train the model
    print(f"Starting fine-tuning for {ASSISTANT_NAME} assistant...")
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")
    
    # Save the model in 4-bit format for efficient inference
    FastLanguageModel.save_pretrained(
        model, tokenizer, OUTPUT_DIR, save_method="safetensors", bits=4
    )
    print(f"Model saved in 4-bit format for efficient inference")

if __name__ == "__main__":
    main()