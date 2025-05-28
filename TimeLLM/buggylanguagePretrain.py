import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, Mamba2Config, GPT2LMHeadModel, Mamba2ForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch

from itertools import count
token_counter = count()  # simple global counter

import sys
#print("sys.path before:", sys.path)
sys.path.insert(0, '/home/nesl/oliver/timeSeriesMamba/mamba_ssm/models/')
#print("sys.path after:", sys.path)
#from mixer_seq_simple import MambaLMHeadModel,MambaTimeHeadModel
sys.path.pop(0)

# Set up argument parser
parser = argparse.ArgumentParser(description="Train GPT2 or Mamba2 on the Pile or OpenWebText dataset")
parser.add_argument("--model_type", type=str, required=True, choices=["gpt2", "mamba2"], help="Type of model to train: 'gpt2' or 'mamba2'")
parser.add_argument("--model_name", type=str, required=True, help="Specific model name or path, e.g., 'gpt2' or 'state-spaces/mamba2-130m'")
parser.add_argument("--dataset", type=str, required=True, choices=["pile", "openwebtext"], help="Dataset to use: 'pile' or 'openwebtext'")
args = parser.parse_args()

# Load the dataset with streaming
dataset_config = {
    "pile": {"name": "monology/pile-uncopyrighted", "split": "train"},
    "openwebtext": {"name": "Skylion007/openwebtext", "split": "train"}
}

dataset_info = dataset_config[args.dataset]
dataset = load_dataset(dataset_info["name"], split=dataset_info["split"], streaming=True)
dataset = dataset.filter(lambda x: len(x["input_ids"]) >= 4)

# Define model configuration based on model_type
model_config = {
    "gpt2": {
        "tokenizer_name": args.model_name,
        "model_class": GPT2LMHeadModel
    },
    "mamba2": {
        "tokenizer_name": "EleutherAI/gpt-neox-20b",
        "model_class": Mamba2ForCausalLM 
    }
}

# Get tokenizer and model class based on model_type
tokenizer_name = model_config[args.model_type]["tokenizer_name"]
model_class = model_config[args.model_type]["model_class"]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token

def tokenize_function(examples):
    tokenized = tokenizer(examples["text"], truncation=True, max_length=1024)
    batch_len = sum(len(input_ids) for input_ids in tokenized["input_ids"])
    for _ in range(batch_len):
        next(token_counter)
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

print(tokenized_dataset[0])  # Should show {"input_ids": [...], "labels": [...], ...}

# Load the model
model = model_class.from_pretrained(args.model_name)

# Set up data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training hyperparameters

total_steps = 1_220_000 #for a more comprehensive training on owt, goodenough is 320k
#total_steps = 27_000_000 #for a more comprehensive training on Pile UnCopyrighted, goodenough is 320k

warmup_steps = int(0.1 * total_steps)  # 10% warmup
batch_size = 8  # Per GPU; effective batch size = 8 × n_gpus

# Set unique output directory to avoid overwriting checkpoints
output_dir = f"./results/{args.dataset}/{args.model_type}/{args.model_name.replace('/', '_')}"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    max_steps=total_steps,
    #num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    save_total_limit=1,
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
print(f"Training {args.model_type} model: {args.model_name} on {args.dataset} dataset")
print(f"Checkpoints will be saved to: {output_dir}")
trainer.train()

print(f"Total tokens processed during streaming: {next(token_counter)}")
