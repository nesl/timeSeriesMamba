from unsloth import FastLanguageModel
from finetune.lib import chat
# from lib import chat
from datetime import datetime
import os 
import pdb 

class Agent:
    def __init__(self, args, model, system_prompt, temperature, top_p):
        self.args = args 
        max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        print(f"using model at: {model}")
        llama_model, tokenizer = FastLanguageModel.from_pretrained(
            # model_name = "./lora_model", # YOUR MODEL YOU USED FOR TRAINING
            model_name = model, # YOUR MODEL YOU USED FOR TRAINING
            # model_name = "unsloth/Llama-3.2-3B-Instruct",
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            )
        FastLanguageModel.for_inference(llama_model) # Enable native 2x faster inference
        self.model = llama_model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p 
        self.tokenizer = tokenizer
        self.chat = [{"role": "user", "content": self.system_prompt
				}]
        
        now = datetime.now()
        current_time_str = now.strftime("%m-%d %H:%M:%S")

        name = f"{args.query}_{args.index}_{args.mode}_{args.eval}_{args.num_trial}_{current_time_str}"
        # name = "llama"
        conv_dir = './conv_history/{}/'.format(model)
        if not os.path.isdir(conv_dir):
            os.makedirs(conv_dir, exist_ok=True)
        self.file_name = conv_dir + name 
        
        if args.mode == 'base':
            self.stop = None
        else:
            self.stop = ["```\n", "```\n\n", "</s>"]
    
    def update(self, content, role):
        self.chat.append(
            {"role": role, "content": content})
    
    def reset(self):
        self.chat = [ {"role": "user", "content": self.system_prompt
				}]

    def step(self, stop=None):
        output = chat(self.model, self.tokenizer, self.chat)[0]
        model_output = self.tokenizer.decode(output)
        model_output = model_output.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        model_output = model_output.split("<|eot_id|>")[0]
        self.chat.append(
            {"role": "assistant", "content": model_output})
        return model_output
    
    def save_chat(self, trial=1, result="None"):
        # save conversation history
        with open(self.file_name + f'_trial_{trial}.txt', 'w') as file:
            # Iterate over each item in the list
            for item in self.chat:
                # Write each item to the file followed by a newline character
                file.write(str(item) + '\n')
            if self.args.target_file is not None:
                file.write(result)


    

if __name__ == '__main__':
    
    args = {}
    llama_agent = Agent(
        # args, model="./Llama-3.2-3B-Instruct", system_prompt="You are a helpful assistant.",
        args, model="unsloth/Llama-3.2-3B-Instruct", system_prompt="You are a helpful assistant.",
        temperature=1, top_p=1
    )
    llama_agent.update(
        content="How to perform FFT in Python?", role="user"
    )
    llama_agent.step()
    pdb.set_trace()


