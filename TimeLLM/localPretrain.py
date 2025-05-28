import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler, GPT2LMHeadModel
from torch.utils.data import IterableDataset, DataLoader
from itertools import count
from tqdm import tqdm

# Use local Mamba implementation
sys.path.insert(0, '/home/nesl/oliver/timeSeriesMamba/mamba_ssm/models/')
from oldmixer_seq_simple import MambaLMHeadModel  # your local model
sys.path.pop(0)

import wandb

import torch.nn as nn



# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, required=True)  # e.g., "state-spaces/mamba2-130m"
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

# Token counter
counter = count()

# Custom iterable dataset
class TokenizedTextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, seq_len):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            tokens = self.tokenizer(
                example["text"],
                return_attention_mask=False,
                return_token_type_ids=False,
                truncation=True,
                max_length=self.seq_len,
            )["input_ids"]
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len:
                chunk = buffer[:self.seq_len]
                buffer = buffer[self.seq_len:]
                yield {"input_ids": torch.tensor(chunk), "labels": torch.tensor(chunk)}
                for _ in chunk:
                    next(counter)

# Model and Tokenizer
if args.model_name == "GPT2Local":
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
else:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaLMHeadModel.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token
model.cuda()
assert model.config.max_position_embeddings >= args.seq_len

#print("tokenizer pad token id: ", tokenizer.pad_token_id)
# Data loader
dataset = TokenizedTextDataset(raw_dataset, tokenizer, args.seq_len)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

# Optimizer + scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=5000, num_training_steps=args.total_steps)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
model.train()
step = 0
pbar = tqdm(total=args.total_steps)

while step < args.total_steps:
    for batch in dataloader:
        batch = {k: v.cuda() for k, v in batch.items()}
        outputs = model(**batch)                    # logits: (B, S, V)
        logits = outputs.logits                     # torch.Tensor
        labels = batch["labels"]                    # (B, S)

        # shift so that token i predicts token i+1
        shift_logits = logits[:, :-1, :].contiguous()      # (B, S-1, V)
        shift_labels = labels[:, 1:].contiguous()          # (B, S-1)

        # flatten
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)), # (B*(S-1), V)
            shift_labels.view(-1)                          # (B*(S-1))
        )
        #print("logit shape, labels shape, loss item: \n", logits.shape, labels.shape, loss.item())
        loss.backward()
        wandb.log({
            "step": step,
            "loss": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "perplexity": torch.exp(loss).item(),
        }, step=step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        step += 1
        pbar.update(1)
        pbar.set_description(f"Step {step} | Loss {loss.item():.4f}")

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
print(f"Training complete. Total tokens processed: {next(counter)}")
