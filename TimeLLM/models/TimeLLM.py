from math import sqrt

import torch
import torch.nn as nn
import torch.nn.init as init

from transformers import AutoModel,MambaModel,AutoTokenizer ,MambaConfig, LlamaConfig, LlamaModel, LlamaForCausalLM, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize

import numpy as np
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from einops import rearrange

import sys
#print("sys.path before:", sys.path)
sys.path.insert(0, '/home/nesl/oliver/timeSeriesMamba/mamba_ssm/models/')
#print("sys.path after:", sys.path)
from mixer_seq_simple import MambaLMHeadModel,MambaTimeHeadModel
sys.path.pop(0)

from huggingface_hub import hf_hub_download

transformers.logging.set_verbosity_error()


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.num_params = configs.num_params
        self.llm_model_name = configs.llm_model
        
        #print("self.num_params in TimeLLM.py: ", self.num_params)
        if configs.llm_model == "Mamba":
            '''
            self.mamba_config = MambaConfig.from_pretrained(f"state-spaces/mamba-{self.num_params}-hf")
            self.mamba_config.num_hidden_layers = configs.llm_layers
            self.mamba_config.output_attentions = True
            self.mamba_config.output_hidden_states = True

            self.llm_model = MambaModel.from_pretrained(
            f"state-spaces/mamba-{self.num_params}-hf",
            config=self.mamba_config
            )
            self.tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba-{self.num_params}-hf")
            '''
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            self.llm_model = MambaLMHeadModel.from_pretrained(f"state-spaces/mamba-{self.num_params}")#, device=device, dtype=dtype)
            
        elif configs.llm_model == "Mamba2":
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            #self.tokenizer = AutoTokenizer.from_pretrained(f"state-spaces/mamba2-{self.num_params}")
            self.llm_model = MambaLMHeadModel.from_pretrained(f"state-spaces/mamba2-{self.num_params}")#, device=device, dtype=dtype)
            #self.llm_model = AutoModel.from_pretrained(f"state-spaces/mamba2-{self.num_params}")
            #print("Mamba2 info: ", self.llm_model.vocab_size)
       
        elif configs.llm_model == "LLAMA3.1":
            model_string = "meta-llama/Meta-Llama-3.1-8B"
            
            self.llama_config = LlamaConfig.from_pretrained(model_string)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            

            self.llm_model = LlamaModel.from_pretrained(
                    #"/home/nesl/oliver/timeSeriesMamba/TimeLLM/Meta-Llama-3.1-8B",
                    model_string,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            tokenizer_config_path = hf_hub_download(repo_id="meta-llama/Meta-Llama-3.1-8B", filename="tokenizer_config.json")
            tokenizer_path = hf_hub_download(repo_id="meta-llama/Meta-Llama-3.1-8B", filename="tokenizer.json")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                    model_string,
                    trust_remote_code=False,
                    local_files_only=True
                )
            
        elif configs.llm_model == "LLAMA3.2":
            model_string = "meta-llama/Llama-3.2-1B"
            self.llama_config = LlamaConfig.from_pretrained(model_string)
            #self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            

            self.llm_model = LlamaModel.from_pretrained(
                    #"/home/nesl/oliver/timeSeriesMamba/TimeLLM/Meta-Llama-3.1-8B",
                    model_string,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config
                    # load_in_4bit=True
                )
            tokenizer_config_path = hf_hub_download(repo_id="meta-llama/Llama-3.2-1B", filename="tokenizer_config.json")
            tokenizer_path = hf_hub_download(repo_id="meta-llama/Llama-3.2-1B", filename="tokenizer.json")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                    model_string,
                    trust_remote_code=False,
                    local_files_only=True
                )

            total_params = 0
            total_el = 0
            #print("Model Parameter Sizes:\n")
            
            for name, param in self.llm_model.named_parameters():
                if param.requires_grad:
                    param_size = param.numel()  # Total number of elements in the parameter
                    #print(f"Parameter: {name}")
                    #print(f" - Shape: {param.shape}")
                    #print(f" - Size: {param_size}\n")
                    total_params += 1
                    total_el += param_size

            #print(f"Total number of named param groups: {total_params}")
            #print(f"Total number of params downloaded off huggingface: {total_el}")

            '''
            print("Testing LLAMA3.2 causal on 'Hey how are you doing?', response: ")
            input_ids = self.tokenizer("Hey how are you doing?", return_tensors="pt")["input_ids"]
            self.llama_config.num_hidden_layers = 2
            
            self.languagellm_model =  LlamaForCausalLM.from_pretrained(
                    model_string,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config)
            out = self.languagellm_model.generate(input_ids, max_new_tokens=10)
            print(self.tokenizer.batch_decode(out))
            '''
        elif configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            '''
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
                '''
            self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_4bit=True
                )
            self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2Pile':
            # Path to your .pt checkpoint file
            checkpoint_path = './results/pile/GPT2Local_320000.pt'  # Adjust to match your .pt file path
            
            # Load the checkpoint from the .pt file
            checkpoint = torch.load(checkpoint_path)
            
            # Extract the config from the checkpoint
            self.gpt2_config = checkpoint['config']
            
            # Modify the config to output attentions and hidden states
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            
            # Initialize the model with the config
            self.llm_model = GPT2Model(self.gpt2_config)
            
            # Load the model's state dictionary from the checkpoint
            self.llm_model.load_state_dict(checkpoint['model'])
            '''
            self.gpt2_config = GPT2Config.from_pretrained('./results/pile/gpt2/gpt2/checkpoint-1220000')  # Path to your checkpoint
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                './results/pile/gpt2/gpt2/checkpoint-10000',  # Path to your checkpoint
                trust_remote_code=True,
                local_files_only=True,  # Use local files since checkpoint is local
                config=self.gpt2_config,
            )
            '''
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',  # Use pretrained tokenizer
                trust_remote_code=True,
                local_files_only=False
            )
            
        elif configs.llm_model == 'GPT2OWT':
            # Path to your .pt checkpoint file
            checkpoint_path = './results/openwebtext/GPT2Local_320000.pt'  # Adjust to match your .pt file path
            
            # Load the checkpoint from the .pt file
            checkpoint = torch.load(checkpoint_path)
            
            # Extract the config from the checkpoint
            self.gpt2_config = checkpoint['config']
            
            # Modify the config to output attentions and hidden states
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            
            # Initialize the model with the config
            self.llm_model = GPT2Model(self.gpt2_config)
            
            # Remap the state dictionary keys by removing 'transformer.' prefix
            checkpoint_state_dict = checkpoint['model']
            remapped_state_dict = {key.replace('transformer.', ''): value for key, value in checkpoint_state_dict.items()}

            # Remove the 'lm_head.weight' key if it exists
            if 'lm_head.weight' in remapped_state_dict:
                del remapped_state_dict['lm_head.weight']
            # Load the remapped state dictionary
            self.llm_model.load_state_dict(remapped_state_dict, strict=True)


            # Load the model's state dictionary from the checkpoint
            #self.llm_model.load_state_dict(checkpoint['model'])
            '''
            self.gpt2_config = GPT2Config.from_pretrained('./results/openwebtext/gpt2/gpt2/checkpoint-1220000')  # Path to your checkpoint
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                './results/openwebtext/gpt2/gpt2/checkpoint-10000',  # Path to your checkpoint
                trust_remote_code=True,
                local_files_only=True,  # Use local files since checkpoint is local
                config=self.gpt2_config,
            )
            '''
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                'openai-community/gpt2',  # Use pretrained tokenizer
                trust_remote_code=True,
                local_files_only=False
            )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained('openai-community/gpt2')

            #self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
        
            '''
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=False,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    'openai-community/gpt2',
                    trust_remote_code=True,
                    local_files_only=False
                )
            '''
        elif configs.llm_model == 'Moirai':
            self.tokenizer = AutoTokenizer.from_pretrained(
                    "meta-llama/Llama-3.2-1B", #shouldn't matter
                    trust_remote_code=False,
                    local_files_only=True
                )
            self.llm_model = Uni2TSWrapper(configs)

        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained('google-bert/bert-base-uncased')

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                #print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )

            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                #print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    'google-bert/bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        '''
        #print("LLM model used is: ", configs.llm_model)
        if configs.rand_init:
            # Reinitialize all parameters with random weights
            for name, param in self.llm_model.named_parameters():
                if param.requires_grad:
                    if "weight" in name:
                        init.normal_(param.data, mean=0.0, std=0.02)  
                    elif "bias" in name:
                        init.constant_(param.data, 0)
            #print("llm weights randomly initialized!")
        '''
        if configs.rand_init:
            torch.manual_seed(configs.init_seed)
            # Reinitialize all parameters with Xavier initialization (activation-agnostic)
            for name, param in self.llm_model.named_parameters():
                if param.requires_grad:
                    if "weight" in name:
                        if param.data.dim() >= 2:  # Check if tensor has 2 or more dimensions
                            init.xavier_normal_(param.data, gain=1.0)
                        else:  # Handle 1D tensors
                            init.normal_(param.data, mean=0.0, std=0.02)  # Fallback to normal initialization
                    elif "bias" in name:
                        init.constant_(param.data, 0)
            torch.manual_seed(configs.seed)


        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        for param in self.llm_model.parameters():
            param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, self.stride, configs.dropout)
        self.num_tokens = 1000
        if self.llm_model_name in ["Moirai","Mamba4Cast"]:
            self.word_embeddings = None
            self.vocab_size = None
            self.mapping_layer = None
        else:
            self.word_embeddings = self.llm_model.get_input_embeddings().weight
            self.vocab_size = self.word_embeddings.shape[0]
            self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)

        self.reprogramming_layer = ReprogrammingLayer(configs.d_model, configs.n_heads, self.d_ff, self.d_llm)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        print("self patch nums: ", self.patch_nums)
        print("self d_ff: ", self.d_ff)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        
        if "embedding_Moirai" in self.llm_model_name:
            
            x_enc = x_enc.permute(0, 2, 1).contiguous()
            enc_out, n_vars = self.patch_embedding(x_enc)
            llama_enc_out = x_enc_out #get rid of all prompting for the TSFM
            dec_out = self.llm_model(x_enc) #YOU SHOULD GET THE EMBEDDINGS HERE FOR TSFM
            #dec_out = self.llm_model(llama_enc_out)
            dec_out = dec_out[:, :, :self.d_ff]
            dec_out = torch.reshape(
                dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
            dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
            #dec_out = dec_out.to(torch.bfloat16)
            dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
            dec_out = dec_out.permute(0, 2, 1).contiguous()
            #print("dec_out shape: ", dec_out.shape)
            dec_out = self.normalize_layers(dec_out, 'denorm')
            return dec_out

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)
       
        
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)

        if "Mamba" not in self.llm_model_name: #i think this is fine, it just feeds embeddings instead of prompts?
            dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        else:
            dec_out = self.llm_model(llama_enc_out).last_hidden_state
        
        #llama enc out is float tensor
        #dec_out = self.llm_model(input_ids=prompt).last_hidden_state
        #dec_out = self.llm_model(input_ids=llama_enc_out, inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        dec_out = dec_out.to(torch.bfloat16)

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        #print("non moirai dec_out shape: ", dec_out.shape)
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags

class Uni2TSWrapper(nn.Module):
    def __init__(self, configs, cov_channel=7, size="base", patch_size="auto", device="cuda"):
        super(Uni2TSWrapper, self).__init__()
        self.device = device
        self.model = MoiraiForecast(
            module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{size}"),
            prediction_length=configs.pred_len,
            context_length=configs.seq_len, #or is it actually llm_dim?
            patch_size=32,
            num_samples=24,#this is the dec_out's first dimension (0)
            target_dim=3,
            feat_dynamic_real_dim=cov_channel,
            past_feat_dynamic_real_dim=None,
        )
        
    def forward(self, data, data_w_cov=None, future_cov=None, use_cov=False):
        #print("moirai forward start: ") #16 is the covariates, and 60 is the seq len. 24 pred len comes from model define
        #print("og data shape: ", data.shape) # Time series values. old Shape: (batch, time, variate)
        #but also they say univariate data is temp_data = data[:,0] so implying the second dim is variates, the first is time
        data = data[0,:].squeeze(-1)
        #data_w_cov = data_w_cov.permute(1, 0, 2).squeeze(-1)  # Shape: (T, num_cov)
        #future_cov = future_cov.permute(1, 0, 2).squeeze(-1)  # Shape: (T_future, num_cov)
        
        #print("squeezed data shape: ", data.shape)
        #print("data_w_cov shape: ", data_w_cov.shape)
        #print("future_cov shape: ", future_cov.shape)

        # Convert to float tensor and handle NaNs
        #print("model dtype: " , next(self.model.parameters()).dtype)
        dtype = next(self.model.parameters()).dtype
        data = torch.tensor(data, dtype=dtype, device=self.device)
        zero_tensor = torch.tensor(0.0, dtype=dtype, device=self.device)
        data = torch.where(torch.isnan(data), zero_tensor, data)


        past_target = rearrange(
            torch.as_tensor(data, dtype=torch.float32), "1 t -> 1 t 1"
            #torch.as_tensor(data, dtype=torch.float32), "t -> 1 t 1"
        )

        past_target = past_target.to(dtype)
        #print("data shape: ", data.shape)
        #print("data type: ", data.dtype)
        

        past_observed_target = torch.ones_like(past_target, dtype=torch.bool, device=self.device)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool, device=self.device).squeeze(-1)
        #print("reshaped past_target shape: ", past_target.shape)
        #print("reshaped past_target dtype: ", past_target.dtype)
        b, t ,n = past_target.shape
        if use_cov:
            covariate_all = []
            for i in range(1, len(future_cov[0]) - 1):
                covariate = torch.cat([data_w_cov[:, i], future_cov[:, i]])
                #print("covariate shape: ", covariate.shape)
                #covariate = torch.tensor(covariate, dtype=self.dtype, device=self.device)
                covariate = rearrange(covariate, "t -> 1 t 1")
                covariate_all.append(covariate)
            
            covariate_all = torch.cat(covariate_all, dim=2)
            observed_covariate = torch.ones_like(covariate_all, dtype=torch.bool, device=self.device)
            '''
            print("past target: ", past_target.shape)
            print("past observed target: ", past_observed_target.shape)
            print("past is pad: ", past_is_pad.shape)
            print("covariate_all: ", covariate_all.shape)
            print("observed_covariate: ", observed_covariate.shape)
            '''
            forecast = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
                feat_dynamic_real=covariate_all,
                observed_feat_dynamic_real=observed_covariate,
            )
        else:
            forecast = self.model(
                past_target=past_target,
                past_observed_target=past_observed_target,
                past_is_pad=past_is_pad,
            )

        # Convert forecast output to PyTorch tensor
        #forecast = torch.tensor(forecast.mean(axis=[0, 1]), device=self.device)
        forecast = torch.tensor(forecast)
        forecast = forecast.permute(1,2,0)
        #print("forecast out shape: ", forecast.shape) #try to get #torch.Size([16, 24, 1]
        return forecast

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape

        scale = 1. / sqrt(E)

        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding