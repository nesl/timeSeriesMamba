import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

def load_and_evaluate_model(checkpoint_path, prompt, max_length=50):
    """
    Load a model checkpoint (Hugging Face directory or .pt file) and evaluate it by completing a prompt.

    Args:
        checkpoint_path (str): Path to the checkpoint (e.g., 'results/openwebtext/GPT2Local_320000.pt')
        prompt (str): Input sentence to complete
        max_length (int): Maximum length of the generated sequence (including prompt)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model and tokenizer from: {checkpoint_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    except Exception as e:
        print(f"Error loading model or tokenizer from directory: {e}")
        print("Attempting to load as a legacy .pt file...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model_config = checkpoint["config"]  # Directly use the config object
            state_dict = checkpoint["model"]
            tokenizer_name = checkpoint["tokenizer"]

            model = GPT2LMHeadModel(model_config)
            model.load_state_dict(state_dict)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            print("Successfully loaded model from legacy .pt file.")
        except Exception as e_legacy:
            print(f"Failed to load as legacy .pt file either: {e_legacy}")
            return None

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated text:", generated_text)
    return generated_text

if __name__ == "__main__":
    #gpt2_checkpoint_path = "results/openwebtext/GPT2Local_320000.pt"
    gpt2_checkpoint_path = "openai-community/gpt2"
    gpt2_checkpoint_path = "results/pile/gpt2_320000.pt"
    prompt = "The trend for this time series data is upwards, suggesting that"
    print("\nEvaluating GPT-2 model from .pt file:")
    load_and_evaluate_model(gpt2_checkpoint_path, prompt)