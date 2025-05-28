import sys
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Config

# Add path to local Mamba implementation (adjust path as needed)
sys.path.insert(0, '/home/nesl/oliver/timeSeriesMamba/mamba_ssm/models/')
from oldmixer_seq_simple import MambaLMHeadModel
sys.path.pop(0)

def load_and_evaluate_model(checkpoint_path, prompt, max_length=50):
    """
    Load a model checkpoint (Mamba or GPT-2) from a .pt file and evaluate it by completing a prompt.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file (e.g., 'checkpoints/model.pt')
        prompt (str): Input sentence to complete
        max_length (int): Maximum length of the generated sequence (including prompt)
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract components
    config = checkpoint["config"]
    state_dict = checkpoint["model"]
    tokenizer_name = checkpoint["tokenizer"]
    
    # Determine model type by inspecting config (assuming config has a 'model_type' field)
    # If your checkpoint doesn't have this, you may need to adjust this logic
    is_gpt2 = hasattr(config, "model_type") and config.model_type == "gpt2"
    
    # Load tokenizer
    if is_gpt2:
        # For GPT-2, use the tokenizer specified in the checkpoint
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Initialize GPT-2 model
        model_config = GPT2Config.from_dict(config.__dict__)
        model = GPT2LMHeadModel(model_config)
    else:
        # For Mamba, use the tokenizer specified in the checkpoint
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # Initialize Mamba model
        model = MambaLMHeadModel(config)
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode and return result
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)
    return generated_text

# Example usage
if __name__ == "__main__":
    # Example checkpoint paths
    mamba_checkpoint_path = "/home/nesl/oliver/timeSeriesMamba/TimeLLM/checkpoints/final-state-spaces_mamba2-130m-pile.pt"
    gpt2_checkpoint_path = "results/pile/gpt2_320000.pt"  # Replace with your GPT-2 .pt checkpoint path
    prompt = "Once upon a time"
    
    #print("Evaluating Mamba model:")
    #load_and_evaluate_model(mamba_checkpoint_path, prompt)
    
    print("\nEvaluating GPT-2 model:")
    load_and_evaluate_model(gpt2_checkpoint_path, prompt)