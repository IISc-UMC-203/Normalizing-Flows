# src/models_img.py
import torch
from torch import nn
import numpy as np
from .layers import BasicBlock, conv3x3, MyBatchNorm2d # Relative imports
from .utils import checkerboard_mask, channel_mask # Relative imports

# --- Backbone for Image Coupling Layers ---
def bottleneck_backbone(in_planes, planes):
    """Creates a ResNet-based backbone for s/t networks."""
    return nn.Sequential(
        conv3x3(in_planes, planes),
        nn.InstanceNorm2d(planes), 
        BasicBlock(planes, planes),
        BasicBlock(planes, planes),
        BasicBlock(planes, planes), 
        BasicBlock(planes, planes), 
        conv3x3(planes, in_planes),
        nn.InstanceNorm2d(in_planes), 
    )

# --- RealNVP Multi-Scale Model for Images ---
class RealNVP_MultiScale(nn.Module):
    """
    Multi-scale RealNVP architecture for image data (MNIST/CIFAR-10).
    Combines logic from the two provided notebooks.
    """
    def __init__(self, input_shape, num_coupling_multiscale=6, num_coupling_final=4, planes=64):
        super().__init__()
        self.input_shape = input_shape # e.g., (1, 28, 28) or (3, 32, 32)
        self.num_coupling_multiscale = num_coupling_multiscale
        self.num_coupling_final = num_coupling_final
        self.initial_planes = planes # Store initial plane count

        self.s_nets = nn.ModuleList()
        self.t_nets = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.s_scale_params = nn.ParameterList()
        self.t_scale_params = nn.ParameterList()
        self.t_bias_params = nn.ParameterList()
        self.shapes = [] # Stores shape at the input of each coupling layer

        current_shape = input_shape
        current_planes = planes

        # --- Multi-scale coupling layers ---
        for i in range(num_coupling_multiscale):
            self.shapes.append(current_shape)
            self.s_nets.append(bottleneck_backbone(current_shape[0], current_planes))
            self.t_nets.append(bottleneck_backbone(current_shape[0], current_planes))
            self.s_scale_params.append(nn.Parameter(torch.zeros(current_shape), requires_grad=True))
            self.t_scale_params.append(nn.Parameter(torch.zeros(current_shape), requires_grad=True))
            self.t_bias_params.append(nn.Parameter(torch.zeros(current_shape), requires_grad=True))
            self.norms.append(MyBatchNorm2d(current_shape[0]))

            if i % 6 == 2: # Squeeze
                new_c = 4 * current_shape[0]
                new_h = current_shape[1] // 2
                new_w = current_shape[2] // 2
                current_shape = (new_c, new_h, new_w)
            if i % 6 == 5: # Factor out
                new_c = current_shape[0] // 2
                current_shape = (new_c, current_shape[1], current_shape[2])
                current_planes = 2 * current_planes # Double planes after factoring out

        # --- Final coupling layers (checkerboard only) ---
        self.final_shape = current_shape
        for i in range(num_coupling_final):
            self.shapes.append(current_shape) # Shape is constant here
            self.s_nets.append(bottleneck_backbone(current_shape[0], current_planes))
            self.t_nets.append(bottleneck_backbone(current_shape[0], current_planes))
            self.s_scale_params.append(nn.Parameter(torch.zeros(current_shape), requires_grad=True))
            self.t_scale_params.append(nn.Parameter(torch.zeros(current_shape), requires_grad=True))
            self.t_bias_params.append(nn.Parameter(torch.zeros(current_shape), requires_grad=True))
            self.norms.append(MyBatchNorm2d(current_shape[0]))

        self.validation = False # Flag for BatchNorm behavior

    def _apply_coupling(self, x, layer_idx, mask):
        """Applies a single coupling transformation."""
        t = self.t_scale_params[layer_idx] * self.t_nets[layer_idx](mask * x) + self.t_bias_params[layer_idx]
        s_raw = self.s_nets[layer_idx](mask * x)
        s = self.s_scale_params[layer_idx] * torch.tanh(s_raw)
        y = mask * x + (1 - mask) * (x * torch.exp(s) + t)
        s_log_det = (1 - mask) * s # Log-determinant contribution from this layer
        return y, s_log_det

    def _invert_coupling(self, y, layer_idx, mask):
        """Inverts a single coupling transformation."""
        t = self.t_scale_params[layer_idx] * self.t_nets[layer_idx](mask * y) + self.t_bias_params[layer_idx]
        s_raw = self.s_nets[layer_idx](mask * y)
        s = self.s_scale_params[layer_idx] * torch.tanh(s_raw)
        x = mask * y + (1 - mask) * ((y - t) * torch.exp(-s))
        return x

    def forward(self, x):
        """Forward pass: image -> latent + log determinant."""
        log_det_s_total = []
        log_det_norm_total = []
        z_factored_out = []
        current_x = x

        # --- Multi-scale Part ---
        for i in range(self.num_coupling_multiscale):
            shape = self.shapes[i]
            # Determine mask: checkerboard for first 3 (0,1,2), channel for next 3 (3,4,5) per block of 6
            is_checkerboard = (i % 6) < 3
            mask = checkerboard_mask(shape, x.device) if is_checkerboard else channel_mask(shape, x.device)
            mask = mask if i % 2 == 0 else (1 - mask) # Alternate inversion

            # Apply coupling
            current_x, s_log_det = self._apply_coupling(current_x, i, mask)
            log_det_s_total.append(torch.flatten(s_log_det, 1)) # Flatten spatial/channel dims

            # Apply BatchNorm
            current_x, norm_log_det = self.norms[i](current_x, validation=self.validation)
            log_det_norm_total.append(norm_log_det) # Already summed spatially in layer

            # Squeeze or Factor out
            if i % 6 == 2: # Squeeze
                current_x = torch.nn.functional.pixel_unshuffle(current_x, 2)
            if i % 6 == 5: # Factor out
                factor_channels = current_x.shape[1] // 2
                z_factored_out.append(torch.flatten(current_x[:, factor_channels:, :, :], 1))
                current_x = current_x[:, :factor_channels, :, :]

        # --- Final Part ---
        final_x = current_x
        for i in range(self.num_coupling_multiscale, self.num_coupling_multiscale + self.num_coupling_final):
            shape = self.shapes[i] # Should be constant here
            mask = checkerboard_mask(shape, final_x.device)
            mask = mask if i % 2 == 0 else (1 - mask) # Alternate inversion

            # Apply coupling
            final_x, s_log_det = self._apply_coupling(final_x, i, mask)
            log_det_s_total.append(torch.flatten(s_log_det, 1))

            # Apply BatchNorm
            final_x, norm_log_det = self.norms[i](final_x, validation=self.validation)
            log_det_norm_total.append(norm_log_det)

        # Combine final tensor with factored-out parts
        z_final = torch.flatten(final_x, 1)
        z_factored_out.append(z_final)
        z_combined = torch.cat(z_factored_out, dim=1)

        # Combine log determinants
        log_det_s_flat = torch.cat(log_det_s_total, dim=1)
        log_det_norm_flat = torch.cat([ld.view(ld.shape[0], -1) for ld in log_det_norm_total], dim=1) if log_det_norm_total else torch.zeros(z_combined.shape[0], 0, device=z_combined.device)

        # Combine all scale parameters for regularization
        all_scales_flat = torch.cat([torch.flatten(p) for p in self.s_scale_params])

        return z_combined, log_det_s_flat, log_det_norm_flat, all_scales_flat


    def inverse(self, z):
        """Inverse pass: latent -> image."""
        if self.training or self.validation:
             raise RuntimeError("Inverse method should only be called in eval mode.")

        # Separate the flattened z into the final part and the factored-out parts
        current_z = z
        z_parts_rev = []
        num_total_layers = self.num_coupling_multiscale + self.num_coupling_final

        # Determine shapes of factored-out parts by reversing the forward logic shapes
        last_final_shape = self.shapes[num_total_layers - 1]
        final_vars = np.prod(last_final_shape)
        z_final_part = torch.reshape(current_z[:, -final_vars:], (-1,) + last_final_shape)
        z_parts_rev.append(z_final_part)
        current_z = current_z[:, :-final_vars]

        for i in reversed(range(self.num_coupling_multiscale)):
             if i % 6 == 5: # Layer where factoring occurred
                 # Shape *before* factoring out was shape[i]
                 prev_shape = self.shapes[i]
                 factored_vars = np.prod(prev_shape) // 2 # Factored out half
                 z_part = torch.reshape(current_z[:, -factored_vars:], (-1,) + (prev_shape[0]//2, prev_shape[1], prev_shape[2]))
                 z_parts_rev.append(z_part)
                 current_z = current_z[:, :-factored_vars]

        assert current_z.shape[1] == 0, "Did not correctly split latent variable z"

        # --- Invert Final Part ---
        current_x = z_parts_rev.pop(0) # Start with the final latent part
        for i in reversed(range(self.num_coupling_multiscale, num_total_layers)):
            shape = self.shapes[i]
            mask = checkerboard_mask(shape, current_x.device)
            mask = mask if i % 2 == 0 else (1 - mask) # Alternate inversion

            # Invert BatchNorm first
            current_x, _ = self.norms[i](current_x, validation=False) # Inverse BN

            # Invert coupling
            current_x = self._invert_coupling(current_x, i, mask)

        # --- Invert Multi-scale Part ---
        for i in reversed(range(self.num_coupling_multiscale)):
            shape = self.shapes[i]
            # Combine with factored-out part *before* inverting this layer
            if i % 6 == 5: # Factoring happened after this layer's forward
                factored_part = z_parts_rev.pop(0)
                current_x = torch.cat((current_x, factored_part), dim=1) # Concatenate channels back

            if i % 6 == 2: # Unsqueeze happened after this layer's forward
                current_x = torch.nn.functional.pixel_shuffle(current_x, 2) # Inverse of unshuffle

            # Invert BatchNorm
            current_x, _ = self.norms[i](current_x, validation=False) # Inverse BN

            # Determine mask
            is_checkerboard = (i % 6) < 3
            mask = checkerboard_mask(shape, current_x.device) if is_checkerboard else channel_mask(shape, current_x.device)
            mask = mask if i % 2 == 0 else (1 - mask) # Alternate inversion

            # Invert coupling
            current_x = self._invert_coupling(current_x, i, mask)


        assert len(z_parts_rev) == 0, "Did not use all factored-out parts"
        # Final result should have the original input shape
        assert current_x.shape[1:] == self.input_shape, f"Output shape {current_x.shape} != input shape {self.input_shape}"

        return current_x

    def validate(self):
        self.eval()
        self.validation = True

    # Override train to reset validation flag
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.validation = False
        return self