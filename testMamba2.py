import torch

batch, length, dim = 2, 64, 512
x = torch.randn(batch, length, dim).to("cuda")
from mamba_ssm import Mamba2
model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")
y = model(x)
print(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}')
print("y shape:", y.shape)
print("x shape:", x.shape)