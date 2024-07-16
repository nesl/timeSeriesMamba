# Copyright (c) 2023, Tri Dao, Albert Gu.

import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoTokenizer, AutoModelForCausalLM

#from mamba_ssm.models.mixer_seq_simple import MambaTimeHeadModel
from models.mixer_seq_simple import MambaTimeHeadModel, MambaLMHeadModel
#from TimeLLM.models.TimeLLM import Model #???

parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

repeats = 10
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
is_mamba = args.model_name.startswith("state-spaces/mamba") or args.model_name.startswith("state-spaces/transformerpp")
if is_mamba:
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    model = MambaTimeHeadModel.from_init(args.model_name, device=device, dtype=dtype)
else:
    print("NOT MAMBA?!")
    #tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    #model = AutoModelForCausalLM.from_init(args.model_name, device_map={"": device}, torch_dtype=dtype)

print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

#training time! we need some data and stuff
model.train()



#this is eval mode, let's train before here
model.eval()
torch.random.manual_seed(0)

if args.prompt is None:
    input_ids = torch.randint(1, 1000, (args.batch, args.promptlen), dtype=torch.long, device="cuda")
    attn_mask = torch.ones_like(input_ids, dtype=torch.long, device="cuda")
else:
    
    tokens = tokenizer(args.prompt, return_tensors="pt")
    input_ids = tokens.input_ids.to(device=device)
    attn_mask = tokens.attention_mask.to(device=device)
    

max_length = input_ids.shape[1] + args.genlen



'''
if is_mamba:
    fn = lambda: model.generate(
        input_ids=input_ids,
        max_length=max_length,
        cg=True,
        return_dict_in_generate=True,
        output_scores=True,
        enable_timing=False,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        min_p=args.minp,
        repetition_penalty=args.repetition_penalty,
    )
else:
    fn = lambda: model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        max_length=max_length,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_k=args.topk,
        top_p=args.topp,
        repetition_penalty=args.repetition_penalty,
    )
'''
class TimeModel(nn.Module):
    def __init__(self, input_dim, genlen, device, dtype):
        super(TimeModel, self).__init__()
        self.genlen = genlen
        self.input_dim = input_dim
        self.outputFN = nn.Sequential(
            nn.Linear(input_dim[1]*input_dim[2], genlen, bias=False, device=device, dtype=dtype),  
            #nn.Linear(genlen, 1, bias=False, device=device, dtype=dtype)  # Transform to [batch_size, 1]
        )
    
    def forward(self, embed_out):
        # Apply the output function
        output = self.outputFN(embed_out.view(self.input_dim[0],self.input_dim[1]*self.input_dim[2]))
        # Ensure the output has the shape [batch_size, genlen, 1]
        output = output.view(self.genlen) #since batch_size is 1, just get rid of it
        return output

embed_in = model.get_input_embeddings()(input_ids)

def fn():
    timeOut = model(embed_in).last_hidden_state
    outputModel = TimeModel(input_dim=timeOut.shape, genlen=args.genlen, device=device, dtype=dtype)
    return outputModel(timeOut)


out = fn()
'''
#THINGS TO TEMPORARILY DEFINE
d_ff = 128
n_vars = args.genlen #???

dec_out = out[:, :, :d_ff]

dec_out = torch.reshape(
    dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
dec_out = dec_out.permute(0, 2, 1).contiguous()

dec_out = self.normalize_layers(dec_out, 'denorm')
'''

if args.prompt is not None:
    #print(tokenizer.batch_decode(out.sequences.tolist()))
    print(out.tolist())
    
torch.cuda.synchronize()

#this repeats section is just for timing how fast it is!
start = time.time()

for _ in range(repeats):
    fn()

torch.cuda.synchronize()
print(f"Prompt length: {len(args.prompt)}, generation length: {args.genlen}")
print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) / repeats * 1000:.0f}ms")
