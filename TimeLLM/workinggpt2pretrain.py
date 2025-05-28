#THIS FILE WORKS FOR A BIT OF PRETRAINIG BUT IT IS ONLY WITH HUGGINGFACE AND DOESN'T USE ENOUGH TRAINING RUNS

import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

# Specify GPUs (optional: set CUDA_VISIBLE_DEVICES externally instead)
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # Use GPU 2 and 3; adjust as needed (e.g., "0" for single GPU)

# Load the dataset with streaming (Pile dataset)
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Tokenize the dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up the model
model = GPT2LMHeadModel.from_pretrained("gpt2")  # ~125M parameters

# Set up data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training hyperparameters
total_steps = 10000
warmup_steps = int(0.1 * total_steps)  # 10% warmup
batch_size = 8  # Per GPU; effective batch size = 8 × n_gpus

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=batch_size,
    max_steps=total_steps,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,  
    learning_rate=3e-3,  
    warmup_steps=warmup_steps,
    weight_decay=0.01,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_epsilon=1e-8,
    lr_scheduler_type="linear",
    fp16=True,  # AMP (torch.autocast)
    report_to="wandb",
    # Multi-GPU settings
    no_cuda=False,  # Ensure CUDA is enabled
    # For DDP, set automatically by torchrun; otherwise, Trainer uses DataParallel
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train the model
trainer.train()