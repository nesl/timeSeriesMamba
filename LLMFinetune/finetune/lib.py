from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from unsloth.chat_templates import standardize_sharegpt
import os
import random
import torch
import numpy as np
import pdb 
from transformers import TextStreamer
from datasets import Dataset
import json
import pandas as pd

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

def prepare_peft_model_n_tokenizer(model_name="unsloth/llama-3-70b-Instruct-bnb-4bit",\
        chat_template="llama-3.1", peft="LoRA"):
    model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if peft == "LoRA":
        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    else:
        raise NotImplementedError("Only LoRA finetuning is supported. Stay tuned!")

    tokenizer = get_chat_template(
    tokenizer,
    chat_template = chat_template,
    )
    return model, tokenizer

def prepare_dataset(tokenizer, dataset_dir=None):
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
        return { "text" : texts, }
    if dataset_dir is None:
        dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
        train_dataset = standardize_sharegpt(dataset)

        dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
        eval_dataset = standardize_sharegpt(dataset)
    else:
        data = read_json(dataset_dir)
        reformated_data = reformat_dict(data)
        data = pd.DataFrame(reformated_data)
        # shuffle data
        data = data.sample(frac=1).reset_index(drop=True)
        # Calculate the split index
        split_index = int(0.95 * len(data))
        train_dataset, eval_dataset = data.iloc[:split_index], data.iloc[split_index:]
        train_dataset = Dataset.from_dict(train_dataset.to_dict(orient="list"))
        train_dataset = standardize_sharegpt(train_dataset)
        
        eval_dataset = Dataset.from_dict(eval_dataset.to_dict(orient="list"))
        eval_dataset = standardize_sharegpt(eval_dataset)


    train_dataset = train_dataset.map(formatting_prompts_func, batched = True,) 
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched = True,) 
    return train_dataset, eval_dataset

def show_memory_stat():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

def load_model_from_dir(dir, inference=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "./lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    if inference:
        FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    return model, tokenizer

def chat(model, tokenizer, messages, text_streamer=None):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")

    if text_streamer is None:
        text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    output = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 2048,
                    use_cache = True, temperature = 1.5, min_p = 0.1)
    return output

def read_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data 
    
def reformat_dict(data):
    # reformat data into {'conversations': [{'from': 'human', 'value': 'How do astronomers determine the original wavelength of light emitted by a celestial body at rest, which is necessary for measuring its speed using the Doppler effect?'}, {'from': 'gpt', 'value': 'Astronom
    reformated_data = []
    for i in range(len(data)):
        entry = {}
        entry['conversations'] = [
            {'from': 'human', 'value': data[i]['user']},
            {'from': 'gpt', 'value': data[i]['assistant']}
        ]
        reformated_data.append(entry)
    return reformated_data

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False