import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, get_scheduler, GPT2LMHeadModel, GPT2Config
from torch.utils.data import IterableDataset, DataLoader, Dataset
from itertools import count
from tqdm import tqdm
import logging
import wandb
import torch.nn as nn
import torch.nn.init as init

# Set up logging
logging.basicConfig(filename='training_errors.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Use local Mamba implementation
sys.path.insert(0, '/home/nesl/oliver/timeSeriesMamba/mamba_ssm/models/')
from oldmixer_seq_simple import MambaLMHeadModel
sys.path.pop(0)

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

# Create validation dataset
validation_texts = [ex["text"] for ex in raw_dataset.take(1000)]
train_raw = raw_dataset.skip(1000)

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
            if not tokens or any(t < 0 or t >= self.tokenizer.vocab_size for t in tokens):
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
else:
    print("error in model name")
    sys.exit(1)
tokenizer.pad_token = tokenizer.eos_token
model.cuda()
# Validation dataset
validation_chunks = []
for text in validation_texts:
    tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False, truncation=True, max_length=None)["input_ids"]
    for i in range(0, len(tokens), args.seq_len):
        chunk = tokens[i:i + args.seq_len]
        if len(chunk) == args.seq_len:
            validation_chunks.append(chunk)

class ValidationDataset(Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.chunks[idx])}

# Data loaders
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
validation_dataset = ValidationDataset(validation_chunks)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, collate_fn=collator)
dataset = TokenizedTextDataset(train_raw, tokenizer, args.seq_len)
dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)
train_iterator = iter(dataloader)

# Optimizer + scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=7000, num_training_steps=args.total_steps)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Validation loss computation
def compute_validation_loss(model, validation_dataloader):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in validation_dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            labels = batch["labels"]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()
            num_batches += 1
    average_loss = total_loss / num_batches
    model.train()
    return average_loss

for name, param in model.named_parameters():
    if param.requires_grad:
        if "weight" in name:
            if param.data.dim() >= 2:  # Check if tensor has 2 or more dimensions
                init.xavier_normal_(param.data, gain=1.0)
            else:  # Handle 1D tensors
                init.normal_(param.data, mean=0.0, std=0.02)  # Fallback to normal initialization
        elif "bias" in name:
            init.constant_(param.data, 0)

# Training loop with early stopping
model.train()
step = 0
pbar = tqdm(total=args.total_steps)
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for step in range(args.total_steps):
    batch = next(train_iterator)
    batch = {k: v.cuda() for k, v in batch.items()}
    
    outputs = model(**batch)
    logits = outputs.logits
    labels = batch["labels"]
    
    if step < 5 or step % 1000 == 0:
        print(f"\n--- Debugging Step {step} ---")
        print("Batch keys:", batch.keys())
        print("Raw input_ids shape:", batch["input_ids"].shape)
        print("Raw labels shape:", batch["labels"].shape)
        print("First sequence input_ids:", batch["input_ids"][0, :10].cpu().numpy())
        print("First sequence labels:", batch["labels"][0, :10].cpu().numpy())
        print("Decoded input (first 20 tokens):", tokenizer.decode(batch["input_ids"][0, :20]))
    
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    
    if (shift_labels < 0).any() or (shift_labels >= model.config.vocab_size).any():
        skipped_batches += 1
        logging.info(f"Skipping batch at step {step} due to invalid labels. Min: {shift_labels.min().item()}, Max: {shift_labels.max().item()}")
        wandb.log({"skipped_batches": skipped_batches, "skipped_examples": skipped_examples}, step=step)
        continue
    
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        skipped_batches += 1
        logging.info(f"Skipping batch at step {step} due to NaN/Inf in logits")
        wandb.log({"skipped_batches": skipped_batches, "skipped_examples": skipped_examples}, step=step)
        continue
    
    loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    wandb.log({
        "step": step,
        "loss": loss.item(),
        "lr": lr_scheduler.get_last_lr()[0],
        "perplexity": torch.exp(loss).item(),
        "skipped_examples": skipped_examples,
        "skipped_batches": skipped_batches,
    }, step=step)

    pbar.update(1)
    pbar.set_description(f"Step {step} | Loss {loss.item():.4f} | Skipped Examples {skipped_examples} | Skipped Batches {skipped_batches}")

    if step % 1000 == 0 and step > 0:
        val_loss = compute_validation_loss(model, validation_dataloader)
        wandb.log({"val_loss": val_loss}, step=step)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at step {step} with validation loss {val_loss}")
                break

    if step % args.save_every == 0 and step > 0:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"results/{args.dataset}/{args.model_name}_{step}.pt"
        torch.save({
            "step": step,
            "model": model.state_dict(),
            "config": model.config,
            "tokenizer": tokenizer.name_or_path,
        }, checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

# Save final model
os.makedirs("checkpoints", exist_ok=True)
final_path = f"results/{args.dataset}/{args.model_name}_{step}.pt"
torch.save({
    "step": step,
    "model": model.state_dict(),
    "config": model.config,
    "tokenizer": tokenizer.name_or_path,
}, final_path)
print(f"Saved final model to: {final_path}")

pbar.close()
print(f"Training complete. Total tokens processed: {next(counter)}, Skipped Examples: {skipped_examples}, Skipped Batches: {skipped_batches}")