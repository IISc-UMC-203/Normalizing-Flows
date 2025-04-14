# src/models_2d.py
import torch
from torch import nn
from .layers import AffineCouplingLayer2D # Relative import

# --- MLP Helper for 2D ---
def create_mlp(input_dim, output_dim, hidden_units=64, depth=3):
    layers = [nn.Linear(input_dim, hidden_units), nn.ReLU()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_units, hidden_units), nn.ReLU()])
    layers.append(nn.Linear(hidden_units, output_dim))
    return nn.Sequential(*layers)

# --- 2D RealNVP Model ---
class RealNVP_2D(nn.Module):
    def __init__(self, dim=2, num_coupling_layers=8, hidden_units=64, depth=3):
        super().__init__()
        assert dim == 2, "This model requires input dimension 2"
        self.dim = dim
        layers = []
        for i in range(num_coupling_layers):
            mask_type = 'checkerboard' if i % 2 == 0 else 'channel' # Alternate masks
            layers.append(AffineCouplingLayer2D(dim, hidden_units, depth, mask_type))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        y = x
        for layer in self.layers:
            y, log_det = layer(y)
            log_det_total += log_det
        return y, log_det_total # Return transformed data and total log determinant

    def inverse(self, z):
        x = z
        for layer in reversed(self.layers): # Apply layers in reverse for inverse
            x = layer.inverse(x)
        return x