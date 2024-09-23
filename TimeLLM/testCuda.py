import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
import torch

# Create a tensor and print its device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
print(tensor)
print(tensor.device)
# Get and print GPU properties
gpu_index = tensor.device.index
gpu_properties = torch.cuda.get_device_properties(gpu_index)
print(f"Using GPU: {gpu_properties.name}")
print(f"GPU Index: {gpu_index}")
print(f"GPU Memory: {gpu_properties.total_memory / (1024 ** 3)} GB")