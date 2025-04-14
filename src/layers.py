# src/layers.py
import torch
from torch import nn
import numpy as np

# --- ResNet Components (Adapted from torchvision) ---

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    # Applying weight norm as seen in CIFAR-10 code snippet
    return nn.utils.weight_norm(nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False, # Bias is False typically in convs followed by BN
        dilation=dilation,
    ))

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    # Applying weight norm
    return nn.utils.weight_norm(nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    ))

class BasicBlock(nn.Module):
    """Basic Residual Block."""
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d # Using InstanceNorm as default from snippets
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# --- Custom BatchNorm ---

class MyBatchNorm2d(nn.modules.batchnorm._NormBase):
    """Custom BatchNorm using running stats during training, with log-det calculation."""
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.005, # Note: Low momentum used 
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        # Manually set affine=False, track_running_stats=True as in original code
        super().__init__(
            num_features, eps, momentum, affine=False, track_running_stats=True, **factory_kwargs
        )

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

    def forward(self, input, validation=False):
        self._check_input_dim(input)
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training:
             # Update running stats using batch stats
            mean = torch.mean(input, dim=(0, 2, 3))
            # Use biased variance for running_var update (matching PyTorch internal)
            var = torch.var(input, dim=(0, 2, 3), unbiased=False)
            with torch.no_grad(): # Update running stats without gradient tracking
                self.running_mean.mul_(1 - exponential_average_factor).add_(mean * exponential_average_factor)
                # Use unbiased var for running var update (matching PyTorch internal comment)
                var_unbiased = torch.var(input, dim=(0, 2, 3), unbiased=True)
                self.running_var.mul_(1 - exponential_average_factor).add_(var_unbiased * exponential_average_factor)

            # Use running stats for the normalization operation *during* training
            current_mean = self.running_mean.view(1, -1, 1, 1).expand_as(input)
            current_var = self.running_var.view(1, -1, 1, 1).expand_as(input)
            denom = (current_var + self.eps).sqrt()
            y = (input - current_mean) / denom
            log_det = -0.5 * torch.log(current_var + self.eps) # Log-det contribution

        elif validation: # Use running stats, no update
            current_mean = self.running_mean.view(1, -1, 1, 1).expand_as(input)
            current_var = self.running_var.view(1, -1, 1, 1).expand_as(input)
            denom = (current_var + self.eps).sqrt()
            y = (input - current_mean) / denom
            log_det = -0.5 * torch.log(current_var + self.eps)

        else: # Inverse operation (eval mode, not validation)
            # Use running stats to reverse normalization
            current_mean = self.running_mean.view(1, -1, 1, 1).expand_as(input)
            current_var = self.running_var.view(1, -1, 1, 1).expand_as(input)
            denom = (current_var + self.eps).sqrt()
            y = input * denom + current_mean
            log_det = -0.5 * torch.log(current_var + self.eps) # Log-det contribution (same sign needed for loss calc)

        # Sum log_det over spatial dimensions and batch, leave channel dim
        log_det_sum = torch.sum(log_det, dim=(2, 3))
        return y, log_det_sum # Return per-channel log-det sum for each batch item

# --- Affine Coupling Layer (for 2D) ---

class AffineCouplingLayer2D(nn.Module):
    """Affine Coupling Layer specifically for 2D data."""
    def __init__(self, dim, hidden_units=64, depth=3, mask_type='checkerboard', scale_factor=1.0):
        super().__init__()
        assert dim == 2, "This layer is designed for dim=2"
        self.dim = dim
        self.mask_type = mask_type
        self.scale_factor = nn.Parameter(torch.tensor(float(scale_factor))) # Ensure float

        # For 2D, dim_keep and dim_transform are always 1
        self.dim_keep = 1
        self.dim_transform = 1

        # Create MLP for s and t networks
        from .models_2d import create_mlp # Relative import
        self.s_net = create_mlp(self.dim_keep, self.dim_transform, hidden_units, depth)
        self.t_net = create_mlp(self.dim_keep, self.dim_transform, hidden_units, depth)

    def _get_mask(self, x):
        # mask_type determines which dimension conditions the other
        if self.mask_type == 'checkerboard':
            # Keep dim 0 (index 0), transform dim 1 (index 1)
            return x[:, :1], x[:, 1:]
        elif self.mask_type == 'channel':
            # Keep dim 1 (index 1), transform dim 0 (index 0)
            return x[:, 1:], x[:, :1]
        else:
            raise ValueError("Unknown mask type")

    def _combine(self, x_keep, x_transform):
        if self.mask_type == 'checkerboard':
            return torch.cat((x_keep, x_transform), dim=1)
        elif self.mask_type == 'channel':
            return torch.cat((x_transform, x_keep), dim=1)

    def forward(self, x):
        x_keep, x_transform = self._get_mask(x)
        s = torch.tanh(self.s_net(x_keep)) * self.scale_factor
        t = self.t_net(x_keep)
        y_transform = x_transform * torch.exp(s) + t
        log_det = torch.sum(s, dim=1) # Sum over the transformed dimension (which is 1)
        y = self._combine(x_keep, y_transform)
        return y, log_det

    def inverse(self, y):
        y_keep, y_transform = self._get_mask(y)
        s = torch.tanh(self.s_net(y_keep)) * self.scale_factor
        t = self.t_net(y_keep)
        x_transform = (y_transform - t) * torch.exp(-s)
        x = self._combine(y_keep, x_transform)
        return x


# --- Helper Modules ---
class Reshape(nn.Module):
    """Reshapes input tensor, keeping batch dimension."""
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple([-1] + list(shape)) # Prepend -1 for batch dim

    def forward(self, x):
        return torch.reshape(x, self.shape)