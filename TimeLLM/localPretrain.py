import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler, GPT2LMHeadModel, GPT2Config
from torch.utils.data import IterableDataset, DataLoader
from itertools import count
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(filename='training_errors.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Use local Mamba implementation
sys.path.insert(0, '/home/nesl/oliver/timeSeriesMamba/mamba_ssm/models/')
from oldmixer_seq_simple import MambaLMHeadModel
sys.path.pop(0)

import wandb
import torch.nn as nn

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--dataset", type=str, choices=["pile", "openwebtext"], required=True)
parser.add_argument("--total_steps", type=int, default=320_000)
parser.add_argument("--save_every", type=int, default=5000)
parser.add_argument("--seq_len", type=int, default=1024)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

wandb.init(
    project="mamba-pretraining",
    name=f"{args.model_name.replace('/', '_')}-{args.dataset}-{args.total_steps}",
    config={
        "model": args.model_name,
        "dataset": args.dataset,
        "total_steps": args.total_steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "lr": 1e-4,
    }
)

# Dataset config
dataset_map = {
    "pile": {"name": "monology/pile-uncopyrighted", "split": "train"},
    "openwebtext": {"name": "Skylion007/openwebtext", "split": "train"},
}
ds_config = dataset_map[args.dataset]

# Load dataset (streaming)
raw_dataset = load_dataset(ds_config["name"], split=ds_config["split"], streaming=True)

# Token counter and skip counters
counter = count()
skipped_examples = 0
skipped_batches = 0

def generate_text(model, tokenizer, prompt, max_length=100, num_return_sequences=1):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
    attention_mask = torch.ones(input_ids.shape, device=input_ids.device)
    output_sequences = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    generated_texts = [tokenizer.decode(seq, skip_special_tokens=True) for seq in output_sequences]
    model.train()
    return generated_texts

# Custom iterable dataset
class TokenizedTextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for i, example in enumerate(self.dataset):
            text = example["text"]
            # Skip non-string, empty, or short text
            if not isinstance(text, str) or not text.strip() or len(text.strip()) < 5:
                global skipped_examples
                skipped_examples += 1
                logging.info(f"Skipping invalid example at index {i}: {text[:100]}...")
                continue
            tokens = self.tokenizer(
                text,
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=self.seq_len,
            )["input_ids"]
            # Skip empty or invalid token lists
            if not tokens or any(t < 0 or t >= self.tokenizer.vocab_size for t in tokens):
                #global skipped_examples
                #skipped_examples += 1
                logging.info(f"Skipping invalid tokens at index {i}: {text[:100]}...")
                continue
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk)}
                for _ in chunk:
                    next(counter)

# Model and Tokenizer
if args.model_name == "GPT2Local":
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    config = GPT2Config.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel(config)
elif args.model_name == "Mamba2":
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained("state-spaces/mamba2-130m")
    #model.resize_token_embeddings(tokenizer.vocab_size)
else:
    print("error in model name")
    sys.exit(1)
tokenizer.pad_token = tokenizer.eos_token
model.cuda()
#assert model.config.max_position_embeddings >= args.seq_len

# Data loader
dataset = TokenizedTextDataset(raw_dataset, tokenizer, args.seq_len)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)#, pad_token_id=tokenizer.pad_token_id)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

# Optimizer + scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=7000, num_training_steps=args.total_steps)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
model.train()
step = 0
pbar = tqdm(total=args.total_steps)

while step < args.total_steps:
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        
        outputs = model(**batch)
        logits = outputs.logits  # (B, S, V)
        labels = batch["labels"]  # (B, S)
        
        # Debugging prints
        if step < 5 or step % 1000 == 0:
            print(f"\n--- Debugging Step {step} ---")
            print("Batch keys:", batch.keys())
            print("Raw input_ids shape:", batch["input_ids"].shape)
            print("Raw labels shape:", batch["labels"].shape)
            print("First sequence input_ids:", batch["input_ids"][0, :10].cpu().numpy())
            print("First sequence labels:", batch["labels"][0, :10].cpu().numpy())
            print("Decoded input (first 20 tokens):", tokenizer.decode(batch["input_ids"][0, :20]))
        
        # Shift logits and labels
        shift_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
        shift_labels = labels[:, 1:].contiguous()      # (B, S-1)
        
        # Replace -100 with pad_token_id and check for invalid labels
        #shift_labels = torch.where(shift_labels == -100, torch.tensor(tokenizer.pad_token_id, device=shift_labels.device), shift_labels)
        if (shift_labels < 0).any() or (shift_labels >= model.config.vocab_size).any():
            #global skipped_batches
            skipped_batches += 1
            logging.info(f"Skipping batch at step {step} due to invalid labels. Min: {shift_labels.min().item()}, Max: {shift_labels.max().item()}")
            wandb.log({"skipped_batches": skipped_batches, "skipped_examples": skipped_examples}, step=step)
            continue  # Skip this batch
        
        # Check for NaN/Inf
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            #global skipped_batches
            skipped_batches += 1
            logging.info(f"Skipping batch at step {step} due to NaN/Inf in logits")
            wandb.log({"skipped_batches": skipped_batches, "skipped_examples": skipped_examples}, step=step)
            continue
        
        # Compute loss
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    logging.info(f"NaN gradient found in {name} at step {step}")
                if param.grad.norm() > 1000:
                    logging.info(f"Large gradient in {name}: {param.grad.norm()} at step {step}")
        
        wandb.log({
            "step": step,
            "loss": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "perplexity": torch.exp(loss).item(),
            "skipped_examples": skipped_examples,
            "skipped_batches": skipped_batches,
        }, step=step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        step += 1
        pbar.update(1)
        pbar.set_description(f"Step {step} | Loss {loss.item():.4f} | Skipped Examples {skipped_examples} | Skipped Batches {skipped_batches}")

        '''
        if step % 5000 == 0 and step > 0:
            print("\n--- Generating text ---")
            prompt = "The quick brown fox"
            generated_texts = generate_text(model, tokenizer, prompt)
            for i, text in enumerate(generated_texts):
                print(f"Generated {i+1}: {text}")
            print("---------------------\n")
        '''
        
        if step == args.total_steps:
            os.makedirs("checkpoints", exist_ok=True)
            final_path = f"results/{args.dataset}/{args.model_name}_{args.total_steps}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "config": model.config,
                "tokenizer": tokenizer.name_or_path,
            }, final_path)
            print(f"Saved final model to: {final_path}")
            break

pbar.close()
print(f"Training complete. Total tokens processed: {next(counter)}, Skipped Examples: {skipped_examples}, Skipped Batches: {skipped_batches}")