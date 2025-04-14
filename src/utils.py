# src/utils.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from tqdm.auto import tqdm
from sklearn.preprocessing import StandardScaler
import sklearn.datasets

# --- Global Mask Caches ---
check_mask_cache = {}
check_mask_device_cache = {}
chan_mask_cache = {}
chan_mask_device_cache = {}

# --- Device Setup ---
def get_device():
    """Gets the best available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")

# --- Masking Functions ---
def checkerboard_mask(shape, device=None):
    """Generates or retrieves a checkerboard mask."""
    global check_mask_cache, check_mask_device_cache
    if device is None:
        device = get_device()

    if shape not in check_mask_cache:
        # Generate mask on CPU first
        mask_np = 1 - np.indices(shape).sum(axis=0) % 2
        check_mask_cache[shape] = torch.from_numpy(mask_np).float() # Store as float

    # Check device-specific cache
    if shape not in check_mask_device_cache or check_mask_device_cache[shape].device != device:
         check_mask_device_cache[shape] = check_mask_cache[shape].to(device)

    return check_mask_device_cache[shape]

def channel_mask(shape, device=None):
    """Generates or retrieves a channel mask (splits channels in half)."""
    global chan_mask_cache, chan_mask_device_cache
    if device is None:
        device = get_device()

    assert len(shape) == 3, f"Channel mask requires 3D shape (C, H, W), got {shape}"
    assert shape[0] % 2 == 0, f"Number of channels must be even for channel mask, got {shape[0]}"

    if shape not in chan_mask_cache:
        # Generate mask on CPU first
        half_channels = shape[0] // 2
        mask_tensor = torch.cat([torch.zeros((half_channels, shape[1], shape[2])),
                                 torch.ones((half_channels, shape[1], shape[2]))],
                                dim=0).float() # Store as float
        assert mask_tensor.shape == shape, (mask_tensor.shape, shape)
        chan_mask_cache[shape] = mask_tensor

    # Check device-specific cache
    if shape not in chan_mask_device_cache or chan_mask_device_cache[shape].device != device:
         chan_mask_device_cache[shape] = chan_mask_cache[shape].to(device)

    return chan_mask_device_cache[shape]

# --- Image Preprocessing ---
def pre_process_img(x, add_noise=True, alpha=0.05):
    """
    Preprocesses image data: dequantization and logit transform.
    """
    y = x
    if add_noise:
        # Add uniform noise U(0, 1/256) for likelihood estimation
        # equivalent to U(0, 1/255) on [0,1] scale)
        noise = torch.rand_like(x) / 255.0
        y = y + noise

    # Logit transform: logit(alpha + (1 - 2*alpha) * y)
    # Ensure input is within valid range for logit
    y = torch.clamp(y, 1e-6, 1.0 - 1e-6) # Clamp slightly inside (0, 1)
    logit_arg = alpha + (1 - 2 * alpha) * y
    logit_arg = torch.clamp(logit_arg, 1e-6, 1.0 - 1e-6)
    y_transformed = torch.logit(logit_arg)
    log_det_per_pixel = -torch.log(logit_arg) - torch.log(1.0 - logit_arg) + np.log(1.0 - 2.0 * alpha)
    log_det_sum = torch.sum(log_det_per_pixel, dim=(1, 2, 3)) # Sum over C, H, W

    return y_transformed, log_det_sum

def post_process_img(y, alpha=0.05):
    """Reverses the logit transform and clamps to [0, 1]."""
    # Inverse logit (sigmoid)
    logit_arg_reconstructed = torch.sigmoid(y)
    # Reverse the inner scaling: x = (logit_arg - alpha) / (1 - 2*alpha)
    x_reconstructed = (logit_arg_reconstructed - alpha) / (1.0 - 2.0 * alpha)
    # Clamp to [0, 1] range
    return torch.clamp(x_reconstructed, min=0.0, max=1.0)

# --- Data Loading Helpers ---
def load_2d_data(dataset_name, n_samples=2000):
    """Loads and scales 2D toy datasets."""
    if dataset_name == 'moons':
        data, _ = sklearn.datasets.make_moons(n_samples=n_samples, noise=0.05)
    elif dataset_name == 'circles':
        data, _ = sklearn.datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    else:
        raise ValueError(f"Unknown 2D dataset name: {dataset_name}")

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data.astype(np.float32), scaler

class Dataset2D(torch.utils.data.Dataset):
    """Simple Dataset wrapper for 2D numpy data."""
    def __init__(self, data):
        self.data = torch.from_numpy(data)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]