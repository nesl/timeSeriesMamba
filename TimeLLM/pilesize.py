from datasets import load_dataset

# Load the dataset with streaming
dataset = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)

# Count the number of examples
count = 0
for example in dataset:
    count += 1
    if count % 10000 == 0:  # Print progress periodically
        print(f"Processed {count} examples")

print(f"Total number of examples: {count}")